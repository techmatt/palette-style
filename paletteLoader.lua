
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

return M
