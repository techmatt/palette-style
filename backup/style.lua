
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

-- Create unique directory for outputs (based on timestamp)
opt.outDir = string.format('%s_%u/', opt.outBaseDir, os.time())
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

if opt.makeNegativeExamples then
    local iLoader = imageLoader.makeImageLoader(opt)
    paletteUtil.generateNegativeImages('savedModels/transformer_iter' .. opt.negativeExamplesIteration .. '.t7', iLoader, opt.negativeExamplesIteration, opt.negativeSamples)
end

if opt.trainTransformer then
    local train = require('trainTransformer')
    local model = models.createModel(opt)
    local iLoader = imageLoader.makeImageLoader(opt)
    
    for i = 1, opt.epochCount do
        train(model, iLoader, opt, i)
    end
end

if opt.trainPaletteChecker then
    local train = require('trainPaletteChecker')
    local model = models.createModel(opt)
    paletteLoader.computePalettes(opt, model, 'images/positives/')
    paletteLoader.computePalettes(opt, model, 'images/negatives/')
    local pLoader = paletteLoader.makePaletteLoader(opt, model)
    
    for i = 1, opt.epochCount do
        train(model, pLoader, opt, i)
    end
end
