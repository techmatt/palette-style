
local threadPool = require('threadPool')
local util = require('util')
local torchUtil = require('torchUtil')

local M = {}

function M.makeImageLoader(opt)
    print('Initializing images from: ' .. opt.imageList)
    local result = {}
    result.donkeys = threadPool.makeThreadPool(opt)
    result.opt = opt
    result.imageList = util.readAllLines(opt.imageList)
    result.positiveList = util.getFileListRecursive(opt.positiveImageList)
    return result
end

local function loadAndResizeImage(path, opt)
    local loadSize = {3, opt.imageSize, opt.imageSize}
    local input = image.load(path, 3, 'float')

    if input:size(2) == loadSize[2] and input:size(3) == loadSize[3] then
        return input
    end
   
    -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
    if input:size(3) < input:size(2) then
       input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
    else
       input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
    end
    return input
end

-- function to load the image, jitter it appropriately (random crops etc.)
local function loadAndCropImage(path, opt)
   local sampleSize = {3, opt.cropSize, opt.cropSize}
   collectgarbage()
   local input = loadAndResizeImage(path, opt)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[3]
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   if iH == oH then h1 = 0 end
   if iW == oW then w1 = 0 end
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(3) == oW)
   assert(out:size(2) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out) end
   return out
end

function M.sampleBatchRandom(imageLoader)
    local opt = imageLoader.opt
    local imageList = imageLoader.imageList
    local donkeys = imageLoader.donkeys

    local RGBImagesCaffe = torch.FloatTensor(opt.transformerBatchSize, 3, opt.cropSize, opt.cropSize)
    
    for b = 1, opt.transformerBatchSize do
        local imageFilename = imageList[ math.random( #imageList ) ]
        donkeys:addjob(
            function()
                local imgRGB = loadAndCropImage(imageFilename, opt)

                imgCaffe = torchUtil.caffePreprocess(imgRGB:clone())
                
                return imgCaffe
            end,
            function(imgCaffe)
                RGBImagesCaffe[b] = imgCaffe
            end)
    end
    donkeys:synchronize()
    
    local batch = {}
    batch.RGBImagesCaffe = RGBImagesCaffe
    return batch
end

function M.sampleBatchPositive(imageLoader, batchSize)
    local opt = imageLoader.opt
    local positiveList = imageLoader.positiveList
    local donkeys = imageLoader.donkeys

    local RGBImagesCaffe = torch.FloatTensor(batchSize, 3, opt.cropSize, opt.cropSize)
    
    for b = 1, batchSize do
        local imageFilename
        imageFilename = positiveList[ math.random( #positiveList ) ]
        donkeys:addjob(
            function()
                local imgRGB = loadAndCropImage(imageFilename, opt)

                imgCaffe = torchUtil.caffePreprocess(imgRGB:clone())
                
                return imgCaffe
            end,
            function(imgCaffe)
                RGBImagesCaffe[b] = imgCaffe
            end)
    end
    donkeys:synchronize()
    
    local batch = {}
    batch.RGBImagesCaffe = RGBImagesCaffe
    return batch
end

return M
