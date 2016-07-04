
local paletteLoader = require('paletteLoader')
local torchUtil = require('torchUtil')

local debugBatchIndices = {[6000]=true, [20000]=true}
-- local debugBatchIndices = {[5]=true}
--local debugBatchIndices = {}

-- Setup a reused optimization state (for adam/sgd).
local optimStatePaletteChecker = {
    learningRate = 0.0
}

local function paramsForEpoch(epoch)
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     1,   1e-4,   0 },
        {  2,     2,   1e-5,   0 },
        {  3,     5,   1e-5,   0 },
        {  6,     10,   1e-5,   0 },
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

-- GPU inputs (preallocate)
local palettes = torch.CudaTensor()
local targetCategories = torch.CudaTensor()

local paletteCheckerParameters, paletteCheckerGradParameters = nil, nil

local function makeCheckerImage(model, opt, img)
    img = torchUtil.caffePreprocess(img)
    local cudaImg = torch.CudaTensor(1, 3, img:size()[2], img:size()[3])
    cudaImg:copy(img)
    model.vggNet:forward(cudaImg)
    local styleData = model.styleLayers[opt.activeStyleLayerName].output:clone()
    
    model.paletteCheckerNet:evaluate()
    local unnormProbs = model.activePaletteChecker:forward(styleData):float()
    local probs = nn.SpatialSoftMax():forward(unnormProbs)
    local positiveProbs = probs:narrow(2, 2, 1)
    --print(positiveProbs:size())
    return positiveProbs[1]
end

local function trainPaletteChecker(model, loader, opt, epoch)
    
    if paletteCheckerParameters == nil then paletteCheckerParameters, paletteCheckerGradParameters = model.paletteCheckerNet:getParameters() end
    
    cutorch.synchronize()

    model.paletteCheckerNet:training()
    
    local dataLoadingTime = 0
    timer:reset()
    
    local classificationLossSum = 0
    local top1Accuracy = -1
    local feval = function(x)
        model.paletteCheckerNet:zeroGradParameters()
        
        for superBatch = 1, opt.paletteSuperBatches do
            local loadTimeStart = dataTimer:time().real
            local batch = paletteLoader.sampleBatch(loader)
            local loadTimeEnd = dataTimer:time().real
            dataLoadingTime = dataLoadingTime + (loadTimeEnd - loadTimeStart)
            
            palettes:resize(batch.palettes:size()):copy(batch.palettes)
            targetCategories:resize(batch.targetCategories:size()):copy(batch.targetCategories)
            
            local dumpPaletteNet = false
            if dumpPaletteNet then
                torchUtil.dumpNet(model.paletteCheckerA, palettes, 'dump/')
            end
            local outputLoss = model.paletteCheckerNet:forward({palettes, targetCategories})
            classificationLossSum = classificationLossSum + outputLoss[1]
            model.paletteCheckerNet:backward({palettes, targetCategories}, outputLoss)
            
            if superBatch == 1 and debugBatchIndices[totalBatchCount] then
                torchUtil.dumpGraph(model.paletteCheckerNet, opt.outDir .. 'paletteCheckerNet' .. totalBatchCount .. '.csv')
            end
            
            if superBatch == 1 then
                --local probs = model.audioNetProbs.data.module.output:clone():exp()
                --top1Accuracy = torchUtil.top1Accuracy(probs, targetCategories)
                --print('category probabilities')
                --print(probs:narrow(1, opt.discriminatorBatchSize / 2 - 4, 8))
            end
        end
        
        return classificationLossSum, paletteCheckerGradParameters
    end
    optim.adam(feval, paletteCheckerParameters, optimStatePaletteChecker)

    if totalBatchCount % 100 == 0 then
        local randomNegativeImage = paletteLoader.randomImage(loader, 1)
        local randomPositiveImage = paletteLoader.randomImage(loader, 2)
        
        local negativeProbs = makeCheckerImage(model, opt, randomNegativeImage)
        local positiveProbs = makeCheckerImage(model, opt, randomPositiveImage)
        
        image.save(opt.outDir .. 'samples/' .. totalBatchCount .. '_negativeImg.jpg', randomNegativeImage)
        image.save(opt.outDir .. 'samples/' .. totalBatchCount .. '_negativeProbs.jpg', negativeProbs)
        
        image.save(opt.outDir .. 'samples/' .. totalBatchCount .. '_positiveImg.jpg', randomPositiveImage)
        image.save(opt.outDir .. 'samples/' .. totalBatchCount .. '_positiveProbs.jpg', positiveProbs)
        
        --local batch = audioLoader.sampleBatch(loader, -1)
        --print(batch.audioClips[5]:size())
        --local waveOut = torchUtil.denormWaveform(batch.audioClips[5][1]):mul(1e8)
        --audio.save(opt.outDir .. 'samples/random' .. totalBatchCount .. '.wav', waveOut, opt.audioRate)
    end
    
    cutorch.synchronize()
    
    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, classificationLossSum,
        optimStatePaletteChecker.learningRate, dataLoadingTime))
    --print('  Accuracy: ' .. top1Accuracy .. '%')
    --print(string.format('  Classification loss: %f', classificationLossSum))
    
    dataTimer:reset()
end

local function train(model, loader, opt, epoch)
    batchNumber = 0

    -- save model
    --this should happen at the end of training, but we keep breaking save so I put it first.
    collectgarbage()

    -- clear the intermediate states in the model before saving to disk
    -- this saves lots of disk space
    model.activePaletteChecker:clearState()
    torch.save(opt.outDir .. 'models/paletteChecker-' .. opt.styleLayers[opt.activeStyleLayerIndex].name .. '-iter' .. opt.negativeExamplesIteration .. '-' .. epoch .. '.t7', model.activePaletteChecker)
        
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)

    local params, newRegime = paramsForEpoch(epoch)
    if newRegime then
        optimStatePaletteChecker = {
        learningRate = params.learningRate,
        weightDecay = params.weightDecay
        }
    end
    cutorch.synchronize()
    
    local tm = torch.Timer()
    
    for i = 1, opt.epochSize do
        batchNumber = batchNumber + 1
        trainPaletteChecker(model, loader, opt, epoch)
        totalBatchCount = totalBatchCount + 1
    end
    
    cutorch.synchronize()

    print('Epoch done')
end

return train
