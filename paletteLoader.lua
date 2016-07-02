
local threadPool = require('threadPool')
local util = require('util')
local torchUtil = require('torchUtil')

local M = {}

function M.computePalettes(opt, model, dir)
    print('Computing palettes in ' .. dir)
    local imageList = util.getFileListRecursive(dir)
    for _, filename in ipairs(imageList) do
        local img = image.load(filename, 3, 'float')
        model.vggNet:forward(img)
        for _, layer in pairs(model.styleLayers) do
            local saveFilename = opt.styleCacheDir .. util.filenameFromPath(filename):gsub('.jpg', '_') .. layer.name .. '.dat'
            
            if not util.fileExists(saveFilename) then
                print('saving ' .. saveFilename .. ' ' .. torchUtil.getSize(layer.output))
                torch.save(saveFilename, layer.output)
            end
        end
    end
end

function M.loadPalettes(opt, model, imageDir)
    local imageList = util.getFileListRecursive(imageDir)
    local result = {}
    for _, filename in ipairs(imageList) do
        local palettes = {}
        for _, layer in pairs(model.styleLayers) do
            local saveFilename = opt.styleCacheDir .. util.filenameFromPath(filename):gsub('.jpg', '_') .. layer.name .. '.dat'
            palettes[layer.name] = torch.load(saveFilename)
            print('loaded ' .. saveFilename .. ' ' .. torchUtil.getSize(palettes[layer.name]))
        end
        table.insert(result, palettes)
    end
end

function M.makePaletteLoader(opt, model)
    print('loading cached palettes')
    local result = {}
    result.donkeys = threadPool.makeThreadPool(opt)
    result.opt = opt
    result.positives = M.loadPalettes(opt, model, 'images/positives')
    result.negatives = M.loadPalettes(opt, model, 'images/negatives')
    
    return result
end

function M.samplePalette(paletteTensor, outTensor, paletteDimension)
    local paletteBorder = (paletteDimension - 1) / 2
    local x = torch.random(1 + paletteBorder, paletteTensor:size()[2] - paletteBorder)
    local y = torch.random(1 + paletteBorder, paletteTensor:size()[3] - paletteBorder)
    local sample = paletteTensor:narrow(2, x, paletteDimension):narrow(3, y, paletteDimension)
    outTensor:copy(sample)
end

function M.sampleBatch(paletteLoader, layerName)
    local opt = paletteLoader.opt
    local positives = paletteLoader.positives
    local negatives = paletteLoader.negatives
    local donkeys = audioLoader.donkeys

    local layerInfo = opt.styleLayers[layerName]
    local palettes = torch.FloatTensor(opt.paletteBatchSize, layerInfo.channels, 5, 5)
    local classLabels = torch.IntTensor(opt.paletteBatchSize)
    
    for b = 1, opt.paletteBatchSize do
        local isNegative = torch.uniform(0.0, 1.0) < opt.negativePaletteRate
        local paletteList
        if isNegative then
            paletteList = negatives
            classLabels[b] = 1
        else
            paletteList = positives
            classLabels[b] = 2
        end
         
        local randomPalette = paletteList[ math.random( #paletteList ) ]
        
        donkeys:addjob(
            function()
                M.samplePalette(randomPalette, palettes[b])
            end,
            function()
                
            end)
    end
    donkeys:synchronize()
    
    local batch = {}
    batch.palettes = palettes
    batch.classLabels = classLabels
    return batch
end

return M
