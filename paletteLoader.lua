
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


return M
