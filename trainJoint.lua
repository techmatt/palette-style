
local imageLoader = require('imageLoader')
local torchUtil = require('torchUtil')

local debugBatchIndices = {[60000]=true, [200000]=true}
-- local debugBatchIndices = {[5]=true}
--local debugBatchIndices = {}

-- Setup a reused optimization state (for adam/sgd).
local optimStateTransformer = {
    learningRate = 0.0
}
local optimStatePalette = {
    learningRate = 0.0
}

local function paramsForEpoch(epoch)
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     1,   1e-3,   0 },
        {  2,     2,   1e-4,   0 },
        {  3,     5,   1e-4,   0 },
        {  6,     10,   1e-4,   0 },
        { 11,     20,   1e-5,   0 },
        { 21,     30,   1e-6,   0 },
        { 31,     40,   1e-7,   0 },
        { 41,    1e8,   1e-7,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- Stuff for logging
local batchNumber               -- Current batch in current epoch
local totalBatchCount = 0       -- Total # of batches across all epochs

local timer = torch.Timer()
local dataTimer = torch.Timer()

local function makeCheckerImage(unnormProbs)
    local probs = nn.SpatialSoftMax():forward(unnormProbs:float())
    local positiveProbs = probs:narrow(1, 2, 1)
    return positiveProbs[1]
end

-- GPU inputs (preallocate)
local RGBImagesCaffe = torch.CudaTensor()
local targetContents = torch.CudaTensor()
local targetCategories1 = torch.CudaTensor()
local targetCategories2 = torch.CudaTensor()

local transformerParameters, transformerGradParameters = nil, nil

local function trainTransformer(model, loader, opt, epoch)
    
    if transformerParameters == nil then transformerParameters, transformerGradParameters = model.styleTransformNet:getParameters() end
    
    cutorch.synchronize()

    targetCategories1:resize(opt.transformerBatchSize, opt.styleLayers[1].finalDim, opt.styleLayers[1].finalDim):fill(2)
    targetCategories2:resize(opt.transformerBatchSize, opt.styleLayers[2].finalDim, opt.styleLayers[2].finalDim):fill(2)
    
    model.styleTransformNet:training()
    --model.paletteCheckers[1]:evaluate()
    --model.paletteCheckers[2]:evaluate()
    
    local dataLoadingTime = 0
    timer:reset()
    
    local contentLossSum, style1LossSum, style2LossSum, totalLossSum = 0, 0, 0, 0
    local top1Accuracy = -1
    local feval = function(x)
        model.styleTransformNet:zeroGradParameters()
        
        for superBatch = 1, opt.transformerSuperBatches do
            local loadTimeStart = dataTimer:time().real
            local batch = imageLoader.sampleBatchRandom(loader)
            local loadTimeEnd = dataTimer:time().real
            dataLoadingTime = dataLoadingTime + (loadTimeEnd - loadTimeStart)
            
            RGBImagesCaffe:resize(batch.RGBImagesCaffe:size()):copy(batch.RGBImagesCaffe)
            
            local targetContentsOut = model.vggContentNet:forward(RGBImagesCaffe)
            targetContents:resize(targetContentsOut:size()):copy(targetContentsOut)
            
            --sourceImage, sourceContent, paletteCategories1, paletteCategories2
            local outputLoss = model.styleTransformNet:forward({RGBImagesCaffe, targetContents, targetCategories1, targetCategories2})
            
            contentLossSum = contentLossSum + outputLoss[1][1]
            style1LossSum = style1LossSum + outputLoss[2][1]
            style2LossSum = style2LossSum + outputLoss[3][1]
            totalLossSum = totalLossSum + outputLoss[1][1] + outputLoss[2][1] + outputLoss[3][1]
            
            model.styleTransformNet:backward({RGBImagesCaffe, targetContents, targetCategories1, targetCategories2}, outputLoss)
            
            if superBatch == 1 and debugBatchIndices[totalBatchCount] then
                torchUtil.dumpGraph(model.styleTransformNet, opt.outDir .. 'styleTransformNet' .. totalBatchCount .. '.csv')
            end
            
            if superBatch == 1 and totalBatchCount % 100 == 0 then
                local inputImage = RGBImagesCaffe[1]:clone()
                inputImage = torchUtil.caffeDeprocess(inputImage)
                
                local outTransformer = model.transformerOutput.data.module.output[1]:clone()
                outTransformer = torchUtil.caffeDeprocess(outTransformer)
                
                local outPalette1 = makeCheckerImage(model.predictedCategories1.data.module.output[1])
                local outPalette2 = makeCheckerImage(model.predictedCategories2.data.module.output[1])
                
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_in.jpg', inputImage)
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_out.jpg', outTransformer)
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_palette1.jpg', outPalette1)
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_palette2.jpg', outPalette2)
            end
        end
        
        model.vggNet:zeroGradParameters()
        model.paletteCheckers[1]:zeroGradParameters()
        model.paletteCheckers[2]:zeroGradParameters()
        
        return totalLossSum, transformerGradParameters
    end
    optim.adam(feval, transformerParameters, optimStateTransformer)

    cutorch.synchronize()
    
    print(('T Epch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, totalLossSum,
        optimStateTransformer.learningRate, dataLoadingTime))
    print(string.format('  Content loss: %f', contentLossSum))
    print(string.format('  Style 1 loss: %f', style1LossSum))
    print(string.format('  Style 2 loss: %f', style2LossSum))
    
    dataTimer:reset()
end

local paletteParameters, paletteGradParameters = nil, nil
local paletteImagesInput = torch.CudaTensor()

local function trainPalette(model, loader, opt, epoch)
    
    if paletteParameters == nil then paletteParameters, paletteGradParameters = model.paletteUpdateNet:getParameters() end
    
    cutorch.synchronize()

    targetCategories1:resize(opt.transformerBatchSize, opt.styleLayers[1].finalDim, opt.styleLayers[1].finalDim):fill(2)
    targetCategories2:resize(opt.transformerBatchSize, opt.styleLayers[2].finalDim, opt.styleLayers[2].finalDim):fill(2)
    
    local sampleTransformedCount = opt.transformerBatchSize / 2
    --local sampleTransformedCount = opt.transformerBatchSize / 4
    local sampleRandomCount = opt.transformerBatchSize / 4
    
    local sampleTransformedStart = 1
    local sampleRandomStart = sampleTransformedStart + sampleTransformedCount
    local samplePositiveStart = sampleRandomStart + sampleRandomCount
    local samplePositiveCount = opt.transformerBatchSize - sampleTransformedCount - sampleRandomCount
    
    --half transformed, quarter random, quarter positive
    targetCategories1:narrow(1, 1, sampleTransformedCount + sampleRandomCount):fill(1)
    targetCategories2:narrow(1, 1, sampleTransformedCount + sampleRandomCount):fill(1)
    
    model.styleTransformNet:evaluate()
    
    local dataLoadingTime = 0
    timer:reset()
    
    local style1LossSum, style2LossSum, totalLossSum = 0, 0, 0
    local feval = function(x)
        model.paletteUpdateNet:zeroGradParameters()
        
        for superBatch = 1, opt.paletteSuperBatches do
            local loadTimeStart = dataTimer:time().real
            local batchRandom = imageLoader.sampleBatchRandom(loader)
            local batchPositive = imageLoader.sampleBatchPositive(loader, samplePositiveCount)
            local loadTimeEnd = dataTimer:time().real
            dataLoadingTime = dataLoadingTime + (loadTimeEnd - loadTimeStart)
            
            RGBImagesCaffe:resize(batchRandom.RGBImagesCaffe:size()):copy(batchRandom.RGBImagesCaffe)
            
            local transformedImages = model.transformer:forward(RGBImagesCaffe)
            paletteImagesInput:resize(transformedImages:size()):copy(transformedImages)
            
            local randomImages = RGBImagesCaffe:narrow(1, sampleRandomStart, sampleRandomCount)
            paletteImagesInput:narrow(1, sampleRandomStart, sampleRandomCount):copy(randomImages)
            paletteImagesInput:narrow(1, samplePositiveStart, samplePositiveCount):copy(batchPositive.RGBImagesCaffe)
            
            --tranformedImages, paletteCategories1, paletteCategories2
            local outputLoss = model.paletteUpdateNet:forward({paletteImagesInput, targetCategories1, targetCategories2})
            
            style1LossSum = style1LossSum + outputLoss[1][1]
            style2LossSum = style2LossSum + outputLoss[2][1]
            totalLossSum = totalLossSum + outputLoss[1][1] + outputLoss[2][1]
            
            model.paletteUpdateNet:backward({paletteImagesInput, targetCategories1, targetCategories2}, outputLoss)
            
            if superBatch == 1 and debugBatchIndices[totalBatchCount] then
                torchUtil.dumpGraph(model.paletteUpdateNet, opt.outDir .. 'paletteUpdateNet' .. totalBatchCount .. '.csv')
            end
            
            if superBatch == 1 and totalBatchCount % 100 == 0 then
                --[[for b = 1, opt.transformerBatchSize do
                    print(targetCategories1[b][5][5])
                    local inputImage = torchUtil.caffeDeprocess(paletteImagesInput[b])
                    image.save(opt.outDir .. 'samples/debug' .. totalBatchCount .. '_' .. b .. '.jpg', inputImage)
                end]]
                
                local testIndices = {sampleTransformedStart, sampleRandomStart, samplePositiveStart}
                for i = 1, 3 do
                    local b = testIndices[i]
                    local img = torchUtil.caffeDeprocess(paletteImagesInput[b])
                    image.save(opt.outDir .. 'samples/palette' .. totalBatchCount .. '_' .. b .. '_img.jpg', img)
                    
                    local outPalette1 = makeCheckerImage(model.predictedCategories1.data.module.output[b])
                    local outPalette2 = makeCheckerImage(model.predictedCategories2.data.module.output[b])
                    
                    image.save(opt.outDir .. 'samples/palette' .. totalBatchCount .. '_' .. b .. '_p1.jpg', outPalette1)
                    image.save(opt.outDir .. 'samples/palette' .. totalBatchCount .. '_' .. b .. '_p2.jpg', outPalette2)
                end
                
                --[[local inputImage = RGBImagesCaffe[1]:clone()
                inputImage = torchUtil.caffeDeprocess(inputImage)
                
                local outTransformer = model.transformerOutput.data.module.output[1]:clone()
                outTransformer = torchUtil.caffeDeprocess(outTransformer)
                
                local outPalette1 = makeCheckerImage(model.predictedCategories1.data.module.output[1])
                local outPalette2 = makeCheckerImage(model.predictedCategories2.data.module.output[1])
                
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_in.jpg', inputImage)
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_out.jpg', outTransformer)
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_palette1.jpg', outPalette1)
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_palette2.jpg', outPalette2)]]
            end
        end
        
        model.vggNet:zeroGradParameters()
        
        return totalLossSum, paletteGradParameters
    end
    optim.adam(feval, paletteParameters, optimStatePalette)

    cutorch.synchronize()
    
    print(('P Epch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, totalLossSum,
        optimStatePalette.learningRate, dataLoadingTime))
    print(string.format('  Style 1 loss: %f', style1LossSum))
    print(string.format('  Style 2 loss: %f', style2LossSum))
    
    dataTimer:reset()
end

local function train(model, loader, opt, epoch)
    batchNumber = 0

    -- save model
    --this should happen at the end of training, but we keep breaking save so I put it first.
    collectgarbage()

    -- clear the intermediate states in the model before saving to disk
    -- this saves lots of disk space
    --model.transformer:clearState()
    --torch.save(opt.outDir .. 'models/transformer' .. epoch .. '.t7', model.transformer)
    
    model.vggNet:clearState()
    model.paletteCheckers[1]:clearState()
    model.paletteCheckers[2]:clearState()
    torch.save(opt.outDir .. 'models/perceptualLoss' .. epoch .. '.t7', model.perceptualLossNet)
    
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)

    local params, newRegime = paramsForEpoch(epoch)
    if newRegime then
        optimStateTransformer = {
            learningRate = params.learningRate,
            weightDecay = params.weightDecay
        }
        optimStatePalette = {
            learningRate = 1e-4,
            weightDecay = params.weightDecay
        }
    end
    cutorch.synchronize()
    
    local tm = torch.Timer()
    
    for i = 1, opt.epochSize do
        batchNumber = batchNumber + 1
        trainTransformer(model, loader, opt, epoch)
        trainPalette(model, loader, opt, epoch)
        totalBatchCount = totalBatchCount + 1
    end
    
    cutorch.synchronize()

    print('Epoch done')
end

return train
