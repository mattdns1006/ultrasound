require "image"
require "gnuplot"
require "nn"
require "cunn"
require "xlua"
require "optim"
require "gnuplot"
dofile("movingAverage.lua")
loadData = require("loadData")
dofile("train.lua")


cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-modelName","deconv1.model","Name of model.")
cmd:option("-modelSave",2000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")
cmd:option("-nThreads",10,"Number of threads.")


cmd:option("-nFeats",16,"Number of features.")

cmd:option("-lr",0.001,"Learning rate.")
cmd:option("-lrDecay",1.1,"Learning rate change factor.")
cmd:option("-lrChange",2000,"How often to change lr.")

cmd:option("-display",1,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",100000,"Number of iterations.")

cmd:option("-ma",100,"Moving average.")
cmd:option("-run",1,"Run.")

cmd:text()
params = cmd:parse(arg)


optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam
dofile("donkeys.lua")

function display(x,y,output,trainOrTest)
	if params.display == 1 then 
		if imgDisplay == nil then 
			local zoom = 4
			local initPic = torch.range(1,torch.pow(100,2),1):reshape(100,100)
			imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay1 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay2 = image.display{image=initPic, zoom=zoom, offscreen=false}

			imgDisplay3 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay4 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay5 = image.display{image=initPic, zoom=zoom, offscreen=false}

			imgDisplay = 1 
		end
		local title
		if trainOrTest == "train" then
			title = "Train"
			image.display{image = x, win = imgDisplay0, legend = title}
			image.display{image = y, win = imgDisplay1, legend = title}
			image.display{image = output, win = imgDisplay2, legend = title}
		else	
			title = "Test"
			image.display{image = x, win = imgDisplay3, legend = title}
			image.display{image = y, win = imgDisplay4, legend = title}
			image.display{image = output, win = imgDisplay5, legend = title}
		end
	end
end



function buildModel()
	local layers = dofile("layers.lua")

	local Convolution = nn.SpatialConvolution
	local Pool = nn.SpatialMaxPooling
	local fmp = nn.SpatialFractionalMaxPooling
	local UpSample = nn.SpatialUpSamplingNearest
	local SBN = nn.SpatialBatchNormalization
	local af = nn.ReLU
	local Linear = nn.Linear
	local Dropout = nn.Dropout
	local nFeats = params.nFeats 
	local function same(model)
		nInputs = nOutputs or 1
		nOutputs = nOutputs or 6 
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		:add(SBN(nOutputs))
		:add(af())
	end
	local function down(model)
		nInputs = nOutputs or 1
		nOutputs = nOutputs or nFeats 
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		:add(SBN(nOutputs))
		:add(af())
		:add(fmp(2,2,0.7,0.7))
		--:add(Pool(3,3,2,2,1,1))
	end
	local function up(model)
		nInputs = nOutputs or 1
		nOutputs = nOutputs or nFeats 
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		:add(SBN(nOutputs))
		:add(af())
		:add(UpSample(2))
	end
		
	local model = nn.Sequential()
	--local testInput = torch.rand(1,3,384,768)
	for i = 1, 12 do down(model); 
	end; for i = 1, 2 do up(model);
	end
	model:add(Convolution(nInputs,1,3,3,1,1,1,1))
	model:add(nn.Sigmoid())
	layers.init(model)

	return model
end
x,y = loadData.loadObs("train")

print("Model name ==>")
print(params.modelName)
if params.loadModel == 1 then
	print("==> Loading model")
	model = torch.load(params.modelName):cuda()
else 	
	model = buildModel():cuda()
end
criterion = nn.MSECriterion():cuda()

function load()
	if i == nil then i = 1 end
	local x,y,output
	x,y = loadData.loadObs("train")

	return x,y
end


function run()
	if i == nil then i = 1 end
	while i < params.nIter do
		donkeys:addjob(function()
					local x,y
					x,y = loadData.loadObs("train")
					return x,y
			       end,
			       function(x,y)
					output,targetResize = train(x,y)
					if i % params.displayFreq == 0 then
						display(x,targetResize,output,"train")
					end
				end
				)
	end
end
if params.run == 1 then run() end
	








