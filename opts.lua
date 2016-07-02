
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
    
    cmd:option('-paletteBatchSize', 256, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-paletteSuperBatches', 1, 'TODO')
    
    cmd:option('-transformerBatchSize', 8, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-transformerSuperBatches', 1, 'TODO')
    
    cmd:option('-imageSize', 256, 'Smallest side of the resized image')
    cmd:option('-cropSize', 224, 'Height and Width of image crop to be used as input layer')
    
    cmd:option('-manualSeed', 2, 'Manually set RNG seed')
    cmd:option('-GPU', 2, 'Default preferred GPU')
    
    cmd:option('-maxVGGDepth', 21, 'TODO')
    cmd:option('-contentLayer', 'relu2_2', 'TODO')
    cmd:option('-styleLayers', {
                                relu3_2={channels=256},
                                relu4_1={channels=512},
                                },'TODO')
    cmd:option('-activeStyleLayer', 'relu4_1', 'The name of the style layer currently being trained')
    cmd:option('-negativePaletteRate', 0.75, 'TODO')
    cmd:option('-paletteDimension', 5, 'TODO')
    
    cmd:option('-contentWeight', 1.0, 'TODO')
    cmd:option('-palette1Weight', 1.0, 'TODO')
    cmd:option('-palette2Weight', 1.0, 'TODO')
    
    
    cmd:option('-trainTransformer', false, 'TODO')
    
    cmd:option('-styleCacheDir', 'styleCache/', 'TODO')
    
    ------------- Training options --------------------
    cmd:option('-epochCount',      100,    'Number of total epochs to run')
    cmd:option('-epochSize',       5000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8, 'number of donkeys to initialize (data loading threads)')
    --cmd:option('-nDonkeys',      0, 'number of donkeys to initialize (data loading threads)')
    
    local opt = cmd:parse(arg or {})
    
    opt.paletteBorder = (opt.paletteDimension - 1) / 2
    
    opt.styleLayersList = {}
    opt.styleLayersList[1] = 'relu3_2'
    opt.styleLayersList[2] = 'relu4_1'
    
    return opt
end

return M
