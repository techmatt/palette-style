
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
    cudnn.convert(vggIn, cudnn)
    vggIn = vggIn:cuda()
    local vggOut = nn.Sequential()
    local vggContentOut = nn.Sequential()
    
    local styleLayers = {}
    local vggLayers = {}
    local contentLayer = nil
    for i = 1, opt.maxVGGDepth do
        local layer = vggIn:get(i)
        local name = layer.name
        
        if contentLayer == nil then
            vggContentOut:add(layer)
        end
        
        --print('layer ' .. i .. ': ' .. name)
        local layerType = torch.type(layer)
        if layer.name == opt.contentLayer then
            print('adding content layer ' .. layer.name)
            contentLayer = layer
        end
        for i = 1, #opt.styleLayers do
            if opt.styleLayers[i].name == layer.name then
                print('adding style layer ' .. layer.name)
                styleLayers[layer.name] = layer
            end
        end
        
        table.insert(vggLayers, layer)
        vggOut:add(layer)
    end
    
    vggIn = nil
    collectgarbage()
    return vggOut, vggContentOut, vggLayers, contentLayer, styleLayers
end

local function addPaletteConv(network,iChannels,oChannels,sizeX,sizeY)
    network:add(cudnn.SpatialConvolution(iChannels,oChannels,sizeX,sizeY,1,1,0,0))
    --network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    network:add(nn.LeakyReLU(true))
end

local function createPaletteChecker128(opt)
    local network = nn.Sequential()

    addPaletteConv(network, 128, 256, 1, 1) -- 128x5x5
    addPaletteConv(network, 256, 256, 3, 1) -- 128x3x5
    addPaletteConv(network, 256, 256, 1, 3) -- 128x3x3
    addPaletteConv(network, 256, 256, 1, 1) -- 128x3x3
    addPaletteConv(network, 256, 256, 3, 1) -- 128x1x3
    addPaletteConv(network, 256, 256, 1, 3) -- 128x1x1
    addPaletteConv(network, 256, 256, 1, 1) -- 128x1x1
    network:add(cudnn.SpatialConvolution(256,2,1,1,1,1,0,0))
    
    --network:add(network, nn.SpatialSoftMax()) -- 2x1x1
    
    return network
end

local function createPaletteChecker256(opt)
    local network = nn.Sequential()

    addPaletteConv(network, 256, 256, 1, 1) -- 256x5x5
    addPaletteConv(network, 256, 256, 3, 1) -- 256x3x5
    addPaletteConv(network, 256, 256, 1, 3) -- 256x3x3
    addPaletteConv(network, 256, 256, 1, 1) -- 256x3x3
    addPaletteConv(network, 256, 256, 3, 1) -- 256x1x3
    addPaletteConv(network, 256, 256, 1, 3) -- 256x1x1
    addPaletteConv(network, 256, 256, 1, 1) -- 256x1x1
    network:add(cudnn.SpatialConvolution(256,2,1,1,1,1,0,0))
    
    --network:add(network, nn.SpatialSoftMax()) -- 2x1x1
    
    return network
end

local function createPaletteChecker512(opt)
    local network = nn.Sequential()

    addPaletteConv(network, 512, 512, 1, 1) -- 512x5x5
    addPaletteConv(network, 512, 512, 3, 1) -- 512x3x5
    addPaletteConv(network, 512, 512, 1, 3) -- 512x3x3
    addPaletteConv(network, 512, 512, 1, 1) -- 512x3x3
    addPaletteConv(network, 512, 512, 3, 1) -- 512x1x3
    addPaletteConv(network, 512, 512, 1, 3) -- 512x1x1
    addPaletteConv(network, 512, 512, 1, 1) -- 512x1x1
    network:add(cudnn.SpatialConvolution(512,2,1,1,1,1,0,0))
    return network
end

local function createPaletteCheckerNet(opt, subnets)
    -- Input nodes
    local palettes = nn.Identity()():annotate({name = 'palettes'})
    local targetCategories = nn.Identity()():annotate({name = 'targetCategories'})

    -- Intermediates
    local classificationOutput = subnets.activePaletteChecker(palettes):annotate({name = 'classificationOutput'})
    
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

local function createTransformer(opt)
    local transformer = nn.Sequential()

    addConvElement(transformer, 3, 32, 7, 1, 3) -- n
    addConvElement(transformer, 32, 64, 3, 2, 1) -- n / 2
    addConvElement(transformer, 64, 128, 3, 2, 1) -- n / 4
    
    addResidualBlock(transformer, 128, 128, 3, 1, 1) -- n / 4
    addResidualBlock(transformer, 128, 128, 3, 1, 1) -- n / 4
    addResidualBlock(transformer, 128, 128, 3, 1, 1) -- n / 4
    
    addUpConvElement(transformer, 128, 64, 3, 2, 1, 1) -- n / 2
    addUpConvElement(transformer, 64, 32, 3, 2, 1, 1) -- n
    transformer:add(cudnn.SpatialConvolution(32,3,3,3,1,1,1,1))

    return transformer
end

local function createStyleNet(opt, subnets, vggLayers)
    -- Input nodes
    local sourceImage = nn.Identity()():annotate({name = 'sourceImage'})
    local sourceContent = nn.Identity()():annotate({name = 'sourceContent'})
    local paletteCategories1 = nn.Identity()():annotate({name = 'paletteCategories1'}) 
    local paletteCategories2 = nn.Identity()():annotate({name = 'paletteCategories2'}) 

    -- Intermediates
    local transformerOutput = subnets.transformer(sourceImage):annotate({name = 'transformerOutput'})
    
    local vggStep = transformerOutput
    local palette1Loss, palette2Loss, contentLoss
    local predictedCategories1, predictedCategories2
    for i, layer in ipairs(vggLayers) do
        vggStep = layer(vggStep):annotate({name = 'vggLayer' .. i .. '_' .. layer.name})
        
        if layer.name == opt.styleLayers[1].name then
            print('adding palette1 loss')
            paletteValues1 = vggStep
            predictedCategories1 = subnets.finalPaletteCheckers[1](vggStep)
            palette1Loss = cudnn.SpatialCrossEntropyCriterion()({predictedCategories1, paletteCategories1}):annotate{name = 'palette1Loss'}
        end
        
        if layer.name == opt.styleLayers[2].name then
            print('adding palette2 loss')
            paletteValues2 = vggStep
            predictedCategories2 = subnets.finalPaletteCheckers[2](vggStep)
            palette2Loss = cudnn.SpatialCrossEntropyCriterion()({predictedCategories2, paletteCategories2}):annotate{name = 'palette2Loss'}
        end
        
        if layer.name == opt.contentLayer then
            print('adding content loss')
            contentLoss = nn.MSECriterion()({vggStep, sourceContent}):annotate{name = 'contentLoss'}
        end
    end
    
    local contentLossMul = nn.MulConstant(opt.contentWeight, true)(contentLoss)
    local palette1LossMul = nn.MulConstant(opt.palette1Weight, true)(palette1Loss)
    local palette2LossMul = nn.MulConstant(opt.palette2Weight, true)(palette2Loss)
    
    -- Full training network including all loss functions
    local styleNet = nn.gModule({sourceImage, sourceContent, paletteCategories1, paletteCategories2},
                                {contentLossMul, palette1LossMul, palette2LossMul})

    cudnn.convert(styleNet, cudnn)
    styleNet = styleNet:cuda()
    graph.dot(styleNet.fg, 'styleNet', 'styleNet')
    return styleNet, transformerOutput, predictedCategories1, predictedCategories2, paletteValues1, paletteValues2
end

local function createPaletteUpdateNet(opt, subnets, vggLayers)
    -- Input nodes
    local sourceImages = nn.Identity()():annotate({name = 'sourceImages'})
    local targetCategories1 = nn.Identity()():annotate({name = 'targetCategories1'}) 
    local targetCategories2 = nn.Identity()():annotate({name = 'targetCategories2'}) 

    local vggStep = transformerOutput
    local palette1Loss, palette2Loss
    local predictedCategories1, predictedCategories2
    for i, layer in ipairs(vggLayers) do
        vggStep = layer(vggStep):annotate({name = 'vggLayer' .. i .. '_' .. layer.name})
        
        if layer.name == opt.styleLayers[1].name then
            print('adding palette1 loss')
            predictedCategories1 = subnets.finalPaletteCheckers[1](vggStep)
            palette1Loss = cudnn.SpatialCrossEntropyCriterion()({predictedCategories1, targetCategories1}):annotate{name = 'palette1Loss'}
        end
        
        if layer.name == opt.styleLayers[2].name then
            print('adding palette2 loss')
            predictedCategories2 = subnets.finalPaletteCheckers[2](vggStep)
            palette2Loss = cudnn.SpatialCrossEntropyCriterion()({predictedCategories2, targetCategories2}):annotate{name = 'palette2Loss'}
        end
    end
    
    local palette1LossMul = nn.MulConstant(opt.palette1Weight, true)(palette1Loss)
    local palette2LossMul = nn.MulConstant(opt.palette2Weight, true)(palette2Loss)
    
    -- Full training network including all loss functions
    local paletteUpdateNet = nn.gModule({sourceImages, targetCategories1, targetCategories2},
                                        {palette1LossMul, palette2LossMul})

    cudnn.convert(paletteUpdateNet, cudnn)
    paletteUpdateNet = paletteUpdateNet:cuda()
    graph.dot(paletteUpdateNet.fg, 'paletteUpdateNet', 'paletteUpdateNet')
    return paletteUpdateNet, predictedCategories1, predictedCategories2
end

local function createModel(opt)
    print('Creating model')

    -- Return table
    local r = {}

    -- Create individual sub-networks
    local subnets = {
        paletteChecker128 = createPaletteChecker128(opt),
        paletteChecker256 = createPaletteChecker256(opt),
        paletteChecker512 = createPaletteChecker512(opt),
        transformer = createTransformer(opt)
    }
    if opt.styleLayers[opt.activeStyleLayerIndex].channels == 128 then
        subnets.activePaletteChecker = subnets.paletteChecker128
    elseif opt.styleLayers[opt.activeStyleLayerIndex].channels == 256 then
        subnets.activePaletteChecker = subnets.paletteChecker256
    elseif opt.styleLayers[opt.activeStyleLayerIndex].channels == 512 then
        subnets.activePaletteChecker = subnets.paletteChecker512
    else
        assert(false, 'palette checker network not defined')
    end
    
    r.transformer = subnets.transformer
    r.activePaletteChecker = subnets.activePaletteChecker
    r.vggNet, r.vggContentNet, r.vggLayers, r.contentLayer, r.styleLayers = createVGG(opt)
    
    -- Create composite nets
    if opt.trainTransformer then
        subnets.finalPaletteCheckers = {}
        r.paletteCheckers = {}
        for i = 1, 2 do
            local filename = 'savedModels/paletteChecker-' .. opt.styleLayers[i].name .. '-iter' .. opt.negativeExamplesIteration .. '.t7'
            subnets.finalPaletteCheckers[i] = torch.load(filename)
            r.paletteCheckers[i] = subnets.finalPaletteCheckers[i]
            print('loaded ' .. filename)
        end
        r.styleNet, r.transformerOutput, r.predictedCategories1, r.predictedCategories2, r.paletteValues1, r.paletteValues2 = createStyleNet(opt, subnets, r.vggLayers)
    elseif opt.trainJoint then
        
    else
        r.paletteCheckerNet = createPaletteCheckerNet(opt, subnets)
    end
    
    collectgarbage()
    
    return r
end


return {
    createModel = createModel
}
