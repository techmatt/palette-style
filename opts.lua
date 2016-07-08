
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Image style transfer using network loss')
    cmd:text()
    cmd:text('Options:')
    
    ------------ Network  options ---------------
    cmd:option('-outBaseDir', 'out', 'TODO')
    cmd:option('-imageList', 'data/imageListCOCO.txt', 'TODO')
    
    cmd:option('-transformerBatchSize', 8, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-transformerSuperBatches', 1, 'TODO')
    cmd:option('-paletteSuperBatches', 1, 'TODO')
    
    
    cmd:option('-imageSize', 256, 'Smallest side of the resized image')
    cmd:option('-cropSize', 224, 'Height and Width of image crop to be used as input layer')
    
    cmd:option('-manualSeed', 2, 'Manually set RNG seed')
    cmd:option('-GPU', 2, 'Default preferred GPU')
    
    cmd:option('-maxVGGDepth', 14, 'TODO')
    cmd:option('-contentLayer', 'relu2_2', 'TODO')
    cmd:option('-styleLayers', { [1] = { name = 'relu2_1', channels = 128, finalDim = 108 }, -- 108
                                 [2] = { name = 'relu3_2', channels = 256, finalDim = 52 }, --maxVGGDepth 14 -- 52
                                 --[1] = { name = 'relu2_2', channels = 128, finalDim = 108 },
                                 --[x] = { name = 'relu4_1', channels = 512, finalDim = 24 }, --maxVGGDepth 21
                                 }, 'TODO')
    cmd:option('-paletteDimension', 5, 'TODO')
    --0.0000001 is too much
    --0.00000001 is sort of okay
    --0.00000003 is still pretty low
    --0.00000005 is still pretty low
    --0.000000075 is good for self-portrait
    --0.000000075 is too high for water color
    cmd:option('-contentWeight', 0.000000002, 'TODO')
    cmd:option('-palette1Weight', 1.0, 'TODO')
    cmd:option('-palette2Weight', 1.0, 'TODO')
    --cmd:option('-TVWeight', 1e-6, 'TODO')
    cmd:option('-TVWeight', 0, 'TODO')
    
    cmd:option('-paletteBatchCutoff', 5000, 'TODO')
    
    --cmd:option('-positiveImageList', 'images/positives/', 'TODO')
    cmd:option('-positiveImageList', 'images/positives-watercolor2/', 'TODO')
    
    cmd:option('-imageIters', 10000, 'TODO')
    
    cmd:option('-runImageOptimization', false, 'TODO')
    cmd:option('-trainTranformer', true, 'TODO')
    
    ------------- Training options --------------------
    cmd:option('-epochCount',      100,    'Number of total epochs to run')
    cmd:option('-epochSize',       2000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8, 'number of donkeys to initialize (data loading threads)')
    --cmd:option('-nDonkeys',      0, 'number of donkeys to initialize (data loading threads)')
    
    local opt = cmd:parse(arg or {})
    
    opt.paletteBorder = (opt.paletteDimension - 1) / 2
    
    return opt
end

return M
