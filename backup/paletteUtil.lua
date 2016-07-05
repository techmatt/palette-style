
local M = {}

local image = require('image')
local imageLoader = require('imageLoader')
local torchUtil = require('torchUtil')

function M.generateNegativeImages(transformerFile, iLoader, transformerIteration, sampleCount)
    print('generating negative examples for iteration ' .. transformerIteration)
    local model = torch.load(transformerFile)
    local RGBImagesCaffe = torch.CudaTensor()
    local outDir = 'images/negatives/' .. 'iter' .. transformerIteration .. '/'
    lfs.mkdir(outDir)
    
    model:evaluate()
    for sample = 1, sampleCount do
        local batch = imageLoader.sampleBatch(iLoader)
        RGBImagesCaffe:resize(batch.RGBImagesCaffe:size()):copy(batch.RGBImagesCaffe)
        local transformedImages = model:forward(RGBImagesCaffe)
        local transformedImage = torchUtil.caffeDeprocess(transformedImages[1])
        
        image.save(outDir .. 'i' .. transformerIteration .. '_neg_' .. sample .. '.jpg', transformedImage)
    end
end

return M
