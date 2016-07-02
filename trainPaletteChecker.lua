
local audioLoader = require('audioLoader')
local torchUtil = require('torchUtil')

local debugBatchIndices = {[6000]=true, [20000]=true}
-- local debugBatchIndices = {[5]=true}
--local debugBatchIndices = {}

-- Setup a reused optimization state (for adam/sgd).
local optimStateAudioNet = {
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
local trainLogger = nil
local batchNumber               -- Current batch in current epoch
local totalBatchCount = 0       -- Total # of batches across all epochs

local timer = torch.Timer()
local dataTimer = torch.Timer()

-- GPU inputs (preallocate)
local audioClips = torch.CudaTensor()
local targetCategories = torch.CudaTensor()

local audioNetParameters, audioNetGradParameters = nil, nil

local function trainAudioNet(model, loader, opt, epoch)
    
    if audioNetParameters == nil then audioNetParameters, audioNetGradParameters = model.audioNet:getParameters() end
    
    cutorch.synchronize()

    model.audioNet:training()
    
    local dataLoadingTime = 0
    timer:reset()
    
    local classificationLossSum = 0
    local top1Accuracy = -1
    local feval = function(x)
        model.audioNet:zeroGradParameters()
        
        for superBatch = 1, opt.audioNetSuperBatches do
            local loadTimeStart = dataTimer:time().real
            local batch = audioLoader.sampleBatch(loader, -1)
            local loadTimeEnd = dataTimer:time().real
            dataLoadingTime = dataLoadingTime + (loadTimeEnd - loadTimeStart)
            
            audioClips:resize(batch.audioClips:size()):copy(batch.audioClips)
            targetCategories:resize(batch.classLabels:size()):copy(batch.classLabels)
            
            local outputLoss = model.audioNet:forward({audioClips, targetCategories})
            classificationLossSum = classificationLossSum + outputLoss[1]
            model.audioNet:backward({audioClips, targetCategories}, outputLoss)
            
            if superBatch == 1 and debugBatchIndices[totalBatchCount] then
                torchUtil.dumpGraph(model.audioNet, opt.outDir .. 'audioNet' .. totalBatchCount .. '.csv')
            end
            
            if superBatch == 1 then
                local probs = model.audioNetProbs.data.module.output:clone():exp()
                top1Accuracy = torchUtil.top1Accuracy(probs, targetCategories)
                --print('category probabilities')
                --print(probs:narrow(1, opt.discriminatorBatchSize / 2 - 4, 8))
            end
        end
        
        return classificationLossSum, audioNetGradParameters
    end
    optim.adam(feval, audioNetParameters, optimStateAudioNet)

    if totalBatchCount % 50 == 0 then
        local batch = audioLoader.sampleBatch(loader, -1)
        --print(batch.audioClips[5]:size())
        local waveOut = torchUtil.denormWaveform(batch.audioClips[5][1]):mul(1e8)
        audio.save(opt.outDir .. 'samples/random' .. totalBatchCount .. '.wav', waveOut, opt.audioRate)
    end
    
    cutorch.synchronize()
    
    print(('DEpoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, classificationLossSum,
        optimStateAudioNet.learningRate, dataLoadingTime))
    print('  Accuracy: ' .. top1Accuracy .. '%')
    print(string.format('  Classification loss: %f', classificationLossSum))
    
    dataTimer:reset()
end

local function train(model, loader, opt, epoch)
    batchNumber = 0

    -- save model
    --this should happen at the end of training, but we keep breaking save so I put it first.
    collectgarbage()

    -- clear the intermediate states in the model before saving to disk
    -- this saves lots of disk space
    model.audioClassifier:clearState()
    torch.save(opt.outDir .. 'models/audioClassifier' .. epoch .. '.t7', model.audioClassifier)
    
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)

    local params, newRegime = paramsForEpoch(epoch)
    if newRegime then
        optimStateAudioNet = {
        learningRate = params.learningRate,
        weightDecay = params.weightDecay
        }
    end
    cutorch.synchronize()
    
    local tm = torch.Timer()
    
    for i = 1, opt.epochSize do
        batchNumber = batchNumber + 1
        trainAudioNet(model, loader, opt, epoch)
        totalBatchCount = totalBatchCount + 1
    end
    
    cutorch.synchronize()

    print('Epoch done')
end

return train
