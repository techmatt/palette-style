
local threadPool = require('threadPool')
local util = require('util')
local torchUtil = require('torchUtil')

local M = {}

function M.computePalettes(opt, model, dir)
    print('Computing palettes in ' .. dir)
    local imageList = util.getFileListRecursive(dir)
    for _, filename in ipairs(imageList) do
        for _, layer in pairs(model.styleLayers) do
            local saveFilename = opt.styleCacheDir .. util.filenameFromPath(filename):gsub('.jpg', '_') .. layer.name .. '.dat'
            
            if not util.fileExists(saveFilename) then
                print('saving ' .. saveFilename .. ' ' .. torchUtil.getSize(layer.output))
                
                local img = image.load(filename, 3, 'float')
                img = torchUtil.caffePreprocess(img):cuda()
                model.vggNet:forward(img)
                
                torch.save(saveFilename, layer.output:float())
            end
        end
    end
end

function M.loadPalettes(opt, model, imageDir)
    local imageList = util.getFileListRecursive(imageDir)
    local result = {}
    for _, imageFilename in ipairs(imageList) do
        local saveFilename = opt.styleCacheDir .. util.filenameFromPath(imageFilename):gsub('.jpg', '_') .. opt.activeStyleLayerName .. '.dat'
        local entry = {
            image = image.load(imageFilename, 3, 'float'),
            palette = torch.load(saveFilename)
            }
        print('loaded ' .. saveFilename .. ' ' .. torchUtil.getSize(entry.palette))
        table.insert(result, entry)
    end
    return result
end

function M.makePaletteLoader(opt, model)
    print('loading cached palettes')
    local result = {}
    result.donkeys = threadPool.makeThreadPool(opt)
    result.opt = opt
    result.positives = M.loadPalettes(opt, model, 'images/positives/')
    result.negatives = M.loadPalettes(opt, model, 'images/negatives/')
    
    return result
end

function M.samplePalette(paletteTensor, outTensor, paletteDimension)
    local x = torch.random(1, paletteTensor:size()[2] - paletteDimension)
    local y = torch.random(1, paletteTensor:size()[3] - paletteDimension)
    
    --print('palette size: ' .. torchUtil.getSize(paletteTensor))
    --print(x)
    --print(y)
    
    local sample = paletteTensor:narrow(2, x, paletteDimension):narrow(3, y, paletteDimension)
    outTensor:copy(sample)
end

function M.randomImage(paletteLoader, category)
    local paletteList
    if category == 1 then
        paletteList = paletteLoader.negatives
    else
        paletteList = paletteLoader.positives
    end
    return paletteList[ math.random( #paletteList ) ].image
end

function M.sampleBatch(paletteLoader)
    local opt = paletteLoader.opt
    local positives = paletteLoader.positives
    local negatives = paletteLoader.negatives
    local donkeys = paletteLoader.donkeys

    local layerInfo = opt.styleLayers[opt.activeStyleLayerIndex]
    local palettes = torch.FloatTensor(opt.paletteBatchSize, layerInfo.channels, opt.paletteDimension, opt.paletteDimension)
    local targetCategories = torch.IntTensor(opt.paletteBatchSize, 1, 1)
    
    for b = 1, opt.paletteBatchSize do
        local isNegative = torch.uniform(0.0, 1.0) < opt.negativePaletteRate
        local paletteList
        if isNegative then
            paletteList = negatives
            targetCategories[b][1][1] = 1
        else
            paletteList = positives
            targetCategories[b][1][1] = 2
        end
         
        local randomPalette = paletteList[ math.random( #paletteList ) ].palette
        
        donkeys:addjob(
            function()
                M.samplePalette(randomPalette, palettes[b], opt.paletteDimension)
            end,
            function()
                
            end)
    end
    donkeys:synchronize()
    
    local batch = {}
    batch.palettes = palettes
    batch.targetCategories = targetCategories
    return batch
end

return M
