local Convolution = nn.SpatialConvolution
local Pool = nn.SpatialMaxPooling
local fmp = nn.SpatialFractionalMaxPooling
local UpSample = nn.SpatialUpSamplingNearest
local SBN = nn.SpatialBatchNormalization
local af = nn.ReLU
local Linear = nn.Linear
local Dropout = nn.Dropout
local layers = dofile("layers.lua")

local models = {}

function models.model1()

	local nFeats = params.nFeats 
	local nFeatsInc = torch.floor(params.nFeats/4)
	local nOutputs
	local nInputs
	local kS = params.kernelSize
	local pad = torch.floor((kS-1)/2)
	local function same(model)
		nInputs = nOutputs or 1
		nOutputs = nOutputs or 6 
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		:add(SBN(nOutputs))
		:add(af())
	end
	local function down(model)
		nInputs = nOutputs or 1
		if nOutputs == nil then
			nOutputs = nFeats
		else 
			nOutputs = nFeatsInc + nOutputs
		end
		model:add(Convolution(nInputs,nOutputs,kS,kS,1,1,pad,pad))
		:add(SBN(nOutputs))
		:add(af())
		:add(fmp(2,2,0.7,0.7))
		--:add(Pool(kS,kS,2,2,1,1))
	end
	local function up(model)
		nInputs = nOutputs or 1
		nOutputs = nOutputs -nFeatsInc
		model:add(Convolution(nInputs,nOutputs,kS,kS,1,1,pad,pad))
		:add(SBN(nOutputs))
		:add(af())
		:add(UpSample(2))
	end
		
	local model = nn.Sequential()
	--local testInput = torch.rand(1,3,384,768)
	for i = 1, params.nDown do down(model); 
	end; for i = 1, params.nUp do up(model);
	end
	nInputs = nOutputs or 1
	model:add(Convolution(nInputs,1,3,3,1,1,1,1))
	model:add(nn.Sigmoid())
	layers.init(model)

	return model
end

return models
