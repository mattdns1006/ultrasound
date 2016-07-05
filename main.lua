require "image"
require "gnuplot"
require "nn"
require "cunn"
require "xlua"
require "optim"
require "gnuplot"
dofile("movingAverage.lua")
dofile("train.lua")
dofile("dice.lua")

cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-modelName","deconv1.model","Name of model.")
cmd:option("-modelSave",2000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")
cmd:option("-nThreads",10,"Number of threads.")
cmd:option("-actualTest",0,"Acutal test predictions.")

cmd:option("-nFeats",16,"Number of features.")
cmd:option("-level",0,"Which level (downsample).")

cmd:option("-lr",0.001,"Learning rate.")
cmd:option("-lrDecay",1.1,"Learning rate change factor.")
cmd:option("-lrChange",2000,"How often to change lr.")

cmd:option("-display",1,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",100000,"Number of iterations.")
cmd:option("-zoom",3,"Image zoom.")

cmd:option("-ma",100,"Moving average.")
cmd:option("-run",1,"Run.")

cmd:option("-nDown",10,"Number of down steps.")
cmd:option("-nUp",3,"Number of up steps.")


cmd:text()

params = cmd:parse(arg)
optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}

optimMethod = optim.adam
loadData = require("loadData")
dofile("donkeys.lua")

function display(x,y,output,trainOrTest,name)
	if params.display == 1 then 
		if imgDisplay == nil then 
			local zoom = params.zoom
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
			image.display{image = x, win = imgDisplay0, legend = title.. " input - " .. name}
			image.display{image = y, win = imgDisplay1, legend = title.. " truth."}
			image.display{image = output, win = imgDisplay2, legend = title.. " prediction."}
		else	
			title = "Test"
			image.display{image = x, win = imgDisplay3, legend = title.. " input - " .. name}
			image.display{image = y, win = imgDisplay4, legend = title.. " truth."}
			image.display{image = output, win = imgDisplay5, legend = title.. " prediction."}
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
	for i = 1, params.nDown do down(model); 
	end; for i = 1, params.nUp do up(model);
	end
	model:add(Convolution(nInputs,1,3,3,1,1,1,1))
	model:add(nn.Sigmoid())
	layers.init(model)

	return model
end


print("Model name ==>")
modelName = string.format("deconv_%d_%d_%d",params.nFeats,params.nDown,params.nUp)
if params.loadModel == 1 then
	print("==> Loading model")
	print(modelName)
	model = torch.load(modelName):cuda()
else 	
	model = buildModel():cuda()
end
outSize = model:forward(torch.rand(1,1,420,580):cuda()):size()
print("==> Output Size")
print(outSize)
criterion = nn.MSECriterion():cuda()

function run()
	if i == nil then 
		i = 1 
		trainMa = MovingAverage.new(params.ma)
		--testMa = MovingAverage.new(params.ma)
		trainDiceMa = MovingAverage.new(params.ma)
		testDiceMa = MovingAverage.new(params.ma)

		trainLosses = {}
		testLosses = {}
		trainDice = {}
		testDice = {}
	end
	while i < params.nIter do
		donkeys:addjob(function()
					local x,y
					if tid == 1 or params.actualTest == 1 then 
						name, x,y = loadData.loadObs("test",imgPaths)
					else
						name, x,y = loadData.loadObs("train",imgPaths)
					end
					return x,tid,name,y
			       end,
			       function(x,tid,name,y)
				        if params.actualTest == 1 then
						pred = model:forward(x)
						predUpscaled = image.scale(pred:squeeze():double(),580,420)
						local name = name:gsub("test/","")
						image.saveJPG("testPredictions/"..name,predUpscaled)
						xlua.progress(i,5508)
						i = i + 1
					else 	
						
						if tid == 1 then
							testOutput, testTarget, testLoss = test(x,y)
							--testLosses[#testLosses+1] = testLoss
							testDice[#testDice+1] = diceCoeff(testOutput,testTarget,0.5) 
							if i % params.displayFreq == 0 then
								display(x,testTarget,testOutput,"test",name)
							end
						else 
							trainOutput, trainTarget, trainLoss = train(x,y)
							trainLosses[#trainLosses+1] = trainLoss
							trainDice[#trainDice+1] = diceCoeff(trainOutput,trainTarget,0.5) 
							if i % params.displayFreq == 0 then
								display(x,trainTarget,trainOutput,"train",name)
							end
						end


						if i % params.ma == 0 and #testDice > params.ma then
							--[[

							local lossesTest = torch.Tensor(testLosses)

							testMA = testMa:forward(lossesTest)
							]]--

							local lossesTrain = torch.Tensor(trainLosses)
							trainMA = trainMa:forward(lossesTrain)

							local trainDiceT = torch.Tensor(trainDice)
							local testDiceT = torch.Tensor(testDice)
							trainMADice = trainDiceMa:forward(trainDiceT)
							testMADice = testDiceMa:forward(testDiceT)
							print(string.format("Model %s has train/test ma (%d) dice scores of {%f,%f) (trLoss = %f)",
									     modelName, params.ma, 
									     trainMADice[{{-1}}]:squeeze(), 
									     testMADice[{{-1}}]:squeeze(),
									     trainMA[{{-1}}]:squeeze()
							)
							)
							collectgarbage()
						end
					end
				end
				)
				if params.actualTest == 1 and i == 5508 then print("Finished testing"); break; end
	end
end
if params.run == 1 then run() end
	










