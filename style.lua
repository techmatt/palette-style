
paths.dofile('globals.lua')

local opts = require('opts')
local opt = opts.parse(arg)

cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')
cudnn.benchmark = true

local util = require('util')
local models = require('models')
local torchUtil = require('torchUtil')
local paletteLoader = require('paletteLoader')
local imageLoader = require('imageLoader')
local paletteUtil = require('paletteUtil')

--util.writeAllLines('data/imageListCOCO.txt', util.getFileListRecursive('/home/mdfisher/data/COCO/'))

print('name the output for this run:')
local response = io.read()

-- Create unique directory for outputs (based on timestamp)
--opt.outDir = string.format('%s_%u/', opt.outBaseDir, os.time())
opt.outDir = string.format('%s_%s/', opt.outBaseDir, response)
print('Saving everything to: ' .. opt.outDir)
lfs.mkdir(opt.outDir)
lfs.mkdir(opt.outDir .. 'models/')
lfs.mkdir(opt.outDir .. 'samples/')
-- Copy over all .lua files
lfs.mkdir(opt.outDir .. 'src/')
for file in lfs.dir('.') do
	if paths.extname(file) == 'lua' then
		os.execute(string.format('cp %s %s/src/%s', file, opt.outDir, file))
	end
end

if opt.trainTranformer then
    local train = require('trainJoint')
    local model = models.createModel(opt)
    local iLoader = imageLoader.makeImageLoader(opt)

    for i = 1, opt.epochCount do
        train(model, iLoader, opt, i)
    end
elseif opt.runImageOptimization then
    local model = models.createModel(opt)
    local perceptualLossNet = torch.load('savedModels/perceptualLoss.t7')
    perceptualLossNet:evaluate()
    
    local optimState = {
      learningRate = 1e1, --1e1?
    }
    
    local contentImage = image.load('images/targets/face.jpg')
    local contentImageCaffe = torchUtil.caffePreprocess(contentImage)
    
    local contentImageBatch = torch.CudaTensor()
    contentImageBatch:resize(1, contentImage:size()[1], contentImage:size()[2], contentImage:size()[3])
    contentImageBatch[1] = contentImageCaffe
    
    local img = torch.randn(contentImage:size()):float():mul(0.001)
    
    local targetContent = model.vggContentNet:forward(contentImageBatch):clone()
    
    local imgCUDA = torch.CudaTensor()
    local targetCategories1 = torch.CudaTensor()
    local targetCategories2 = torch.CudaTensor()
    imgCUDA:resize(1, contentImage:size()[1], contentImage:size()[2], contentImage:size()[3])
    
    targetCategories1:resize(1, 188, 145):fill(2)
    targetCategories2:resize(1, 92, 71):fill(2)
    
    local totalCalls = 0
    local function feval(x)
        
        imgCUDA[1] = img
        
        local outputLoss = perceptualLossNet:forward({imgCUDA, targetContent, targetCategories1, targetCategories2})
    
        local contentLoss = outputLoss[1][1]
        local style1Loss = outputLoss[2][1]
        local style2Loss = outputLoss[3][1]
        local loss = contentLoss + style1Loss + style2Loss
        
        --outputLoss[1][1] = outputLoss[1][1] * 0.0001
        
        perceptualLossNet:backward({img, targetContent, targetCategories1, targetCategories2}, outputLoss)
        
        local grad = perceptualLossNet.gradInput[1]:float()
        --local grad = net:updateGradInput(x, dy)
        
        if totalCalls % 50 == 0 then
            print('iter ' .. totalCalls)
            print('content loss: ' .. contentLoss)
            print('style 1 loss: ' .. style1Loss)
            print('style 2 loss: ' .. style2Loss)
            --print(img:size())
            local imgSave = torchUtil.caffeDeprocess(img:cuda())
            image.save(opt.outDir .. 'img_' .. totalCalls .. '.jpg', imgSave)
        end
        totalCalls = totalCalls + 1
        
        return loss, grad
    end
    
    for t = 1, opt.imageIters do
        local x, losses = optim.adam(feval, img, optimState)
    end
end