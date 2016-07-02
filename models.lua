
require('nnModules')

local useResidualBlock = true
local useBatchNorm = true

local function addConvElement(network,iChannels,oChannels,size,stride,padding)
    network:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    if useBatchNorm then network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3)) end
    network:add(cudnn.ReLU(true))
end

local function addLinearElement(network,iChannels,oChannels)
    network:add(nn.Linear(iChannels, oChannels))
    if useBatchNorm then network:add(cudnn.BatchNormalization(oChannels, 1e-3)) end
    network:add(cudnn.ReLU(true))
end

local function addUpConvElement(network,iChannels,oChannels,size,stride,padding,extra)
    network:add(cudnn.SpatialFullConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding,extra,extra))
    --network:add(nn.SpatialUpSamplingNearest(stride))
    --network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,1,1,padding,padding))
    if useBatchNorm then network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3)) end
    network:add(cudnn.ReLU(true))
end

local function addResidualBlock(network,iChannels,oChannels,size,stride,padding)
    --addConvElement(network,iChannels,oChannels,size,stride,padding)
    --addConvElement(network,iChannels,oChannels,size,stride,padding)

    local s = nn.Sequential()
        
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    if useBatchNorm then s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3)) end
    s:add(cudnn.ReLU(true))
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    if useBatchNorm then s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3)) end
    
    if useResidualBlock then
        --local shortcut = nn.narrow(3, )
        
        local block = nn.Sequential()
            :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
            :add(nn.CAddTable(true))
        network:add(block)
    else
        s:add(nn.ReLU(true))
        network:add(s)
    end
end

local function createVGG(opt)
    local vggIn = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt',
                                 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn'):float()
    local vggOut = nn.Sequential()
    
    local styleLayers = {}
    local contentLayer = nil
    for i = 1, opt.maxVGGDepth do
        local layer = vggIn:get(i)
        local name = layer.name
        print('layer ' .. i .. ': ' .. name)
        local layerType = torch.type(layer)
        if layer.name == opt.contentLayer then
            print('adding content layer ' .. layer.name)
            contentLayer = layer
        end
        if opt.styleLayers[layer.name] then
            print('adding style layer ' .. layer.name)
            table.insert(styleLayers, layer)
        end
        vggOut:add(layer)
    end
    
    vggIn = nil
    collectgarbage()
    return vggOut, contentLayer, styleLayers
end

local function createPaletteCheckerA(opt)
    local network = nn.Sequential()

    addConvElement(network, 256, 256, 1, 1, 0) -- 256x5x5
    addConvElement(network, 256, 128, 5, 1, 0) -- 128x5x5
    addConvElement(network, 128, 128, 3, 1, 0) -- 128x3x3
    addConvElement(network, 128, 128, 3, 1, 0) -- 128x1x1
    addConvElement(network, 128, 2, 1, 1, 0)   -- 2x1x1
    
    --network:add(network, nn.SpatialSoftMax()) -- 2x1x1
    
    return network
end

local function createPaletteCheckerNet(opt, subnets)
    -- Input nodes
    local palettes = nn.Identity()():annotate({name = 'palettes'})
    local targetCategories = nn.Identity()():annotate({name = 'targetCategories'})

    -- Intermediates
    local classificationOutput = subnets.paletteCheckerA(palettes):annotate({name = 'classificationOutput'})
    
    print('adding class loss')
    --local classLoss = nn.MSELoss()({classificationOutput, targetCategories}):annotate{name = 'classLoss'}
    local classLoss = cudnn.SpatialCrossEntropyCriterion()({classificationOutput, targetCategories}):annotate{name = 'classLoss'}
    
    -- Full training network including all loss functions
    local paletteCheckerNet = nn.gModule({palettes, targetCategories}, {classLoss})

    cudnn.convert(paletteCheckerNet, cudnn)
    paletteCheckerNet = paletteCheckerNet:cuda()
    graph.dot(paletteCheckerNet.fg, 'paletteChecker', 'paletteChecker')
    return paletteCheckerNet, classificationOutput
end

local function createModel(opt)
    print('Creating model')

    -- Return table
    local r = {}

    -- Create individual sub-networks
    local subnets = {
        paletteCheckerA = createPaletteCheckerA(opt),
    }
    r.paletteCheckerA = subnets.paletteCheckerA
    r.vggNet, r.contentLayer, r.styleLayers = createVGG(opt)
    
    -- Create composite nets
    r.paletteCheckerNet = createPaletteCheckerNet(opt, subnets)
    
    collectgarbage()
    
    return r
end


return {
    createModel = createModel
}
