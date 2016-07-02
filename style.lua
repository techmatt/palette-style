
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
--local imageLoader = require('imageLoader')
--local train = require('trainThumbnailUpsampler')

local model = models.createModel(opt)
--paletteLoader.computePalettes(opt, model, 'images/positives/')
--paletteLoader.computePalettes(opt, model, 'images/negatives/')

local pLoader = paletteLoader.makePaletteLoader(opt, model)

do return end
--local imgLoader = imageLoader.makeImageLoader(opt)

--torchUtil.vibrancyTest(imgLoader.imageLists, 500, 'vibrancyTest/')

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

for i=1,opt.epochCount do
   train(model, imgLoader, opt, i)
end
