
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Image style transfer using network loss')
    cmd:text()
    cmd:text('Options:')
    
    ------------ Network  options ---------------
    cmd:option('-outBaseDir', 'out', 'TODO')
    cmd:option('-imageListBase', 'data/places', 'TODO')
    cmd:option('-batchSize', 64, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-superBatches', 2, 'TODO')
    cmd:option('-imageSize', 256, 'Smallest side of the resized image')
    cmd:option('-cropSize', 224, 'Height and Width of image crop to be used as input layer')
    
    cmd:option('-manualSeed', 2, 'Manually set RNG seed')
    cmd:option('-GPU', 2, 'Default preferred GPU')
    
    cmd:option('-maxVGGDepth', 21, 'TODO')
    cmd:option('-contentLayer', 'relu2_2', 'TODO')
    cmd:option('-styleLayers', {['relu3_2']=true, ['relu4_1']=true}, 'TODO')
    
    cmd:option('-styleCacheDir', 'styleCache/', 'TODO')
    
    ------------- Training options --------------------
    cmd:option('-epochCount',      100,    'Number of total epochs to run')
    cmd:option('-epochSize',       5000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8, 'number of donkeys to initialize (data loading threads)')
    --cmd:option('-nDonkeys',      0, 'number of donkeys to initialize (data loading threads)')
    
    local opt = cmd:parse(arg or {})
    return opt
end

return M
