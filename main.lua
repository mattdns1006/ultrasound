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
dofile("counter.lua")

cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-modelName","deconv1.model","Name of model.")
cmd:option("-modelSave",5000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")
cmd:option("-nThreads",10,"Number of threads.")
cmd:option("-trainAll",0,"Train on all images in training set.")
cmd:option("-actualTest",0,"Acutal test predictions.")

cmd:option("-nFeats",16,"Number of features.")
cmd:option("-kernelSize",3,"Kernel size.")
cmd:option("-level",0,"Which level (downsample).")
cmd:option("-diceThreshold",0.5,"What threshold?")

cmd:option("-lr",0.001,"Learning rate.")
cmd:option("-lrDecay",1.1,"Learning rate change factor.")
cmd:option("-lrChange",5000,"How often to change lr.")

cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",200000,"Number of iterations.")
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
models = require "models"


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

print("Model name ==>")
modelName = string.format("deconv_%d_%d_%d",params.nFeats,params.nDown,params.nUp)
if params.loadModel == 1 then
	print("==> Loading model")
	print(modelName)
	model = torch.load(modelName):cuda()
else 	
	model = models.model2():cuda()
end
print(model)
local sf = 1/torch.pow(2,params.level)
params.inSize = torch.rand(1,1,420*sf,580*sf):size()
params.outSize = model:forward(torch.rand(1,1,420*sf,580*sf):cuda()):size()
print("==> Input Size")
print(params.inSize)
print("==> Output Size")
print(params.outSize)
criterion = nn.MSECriterion():cuda()
print("==> Init threads")
dofile("donkeys.lua")

function run()
	if i == nil then 
		i = 1 
		trainDiceMa = MovingAverage.new(params.ma)
		testDiceMa = MovingAverage.new(params.ma)
		trainLossMa = MovingAverage.new(params.ma)

		trainCounter = Counter.new()
		testCounter = Counter.new()

		trainLosses = {}
		trainDice = {}
		testDice = {}
		testDice2 = {}
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
						local name = name:gsub("test/"..params.level.."/","")
						image.saveJPG("testPredictions/"..name,predUpscaled)
						xlua.progress(i,5508)
						i = i + 1
					else 	
						
						if tid == 1 and params.trainAll == 0 then
							testCounter:add(name)
							testOutput, testTarget, testLoss = test(x,y)
							--testDice[#testDice+1] = diceROC(testOutput,testTarget) 
							testDice2[#testDice2+1] = diceCoeff(testOutput,testTarget,params.diceThreshold) 

							if i % params.displayFreq == 0 and params.display == 1 then
								--[[

								local testDiceT = torch.Tensor(testDice):mean(1):squeeze()
								local X = torch.linspace(0.05,0.95,testDiceT:size(1))
								local max = X[testDiceT:eq(testDiceT:max())][1]
								local title = string.format("Max = %f",max)
								gnuplot.plot({title,X,testDiceT})
								]]--
								display(x,testTarget,testOutput,"test",name)
							end
						else 
							trainCounter:add(name)
							trainOutput, trainTarget, trainLoss = train(x,y)
							trainLosses[#trainLosses + 1] = trainLoss
							trainDice[#trainDice+1] = diceCoeff(trainOutput,trainTarget,params.diceThreshold) 
							if i % params.displayFreq == 0 and params.display == 1 then
								--display(x,trainTarget,trainOutput,"train",name)
							end
						end

						if i % params.ma == 0 and #testDice2 > params.ma then

							local trainLossesT = torch.Tensor(trainLosses)
							local trainDiceT = torch.Tensor(trainDice)
							local testDiceT = torch.Tensor(testDice2)
							local trainMALosses = trainLossMa:forward(trainLossesT)
							local trainMADice = trainDiceMa:forward(trainDiceT)
							local testMADice = testDiceMa:forward(testDiceT)
							print(string.format("Model %s has train/test ma (%d) dice scores of {%f,%f} (trL = {%f}). ",
									     modelName, params.ma, 
									     trainMADice[{{-1}}]:squeeze(), 
									     testMADice[{{-1}}]:squeeze(),
									     trainMALosses[{{-1}}]:squeeze()

							)
							)
							trainLosses = {}
							trainDice = {}
							testDice = {}
							testDice2 = {}
							collectgarbage()
						end
						if i % 10000 ==0  then
							print("Number of different train/test observations seen",csv.length(trainCounter),csv.length(testCounter))
						end
					end
				end
				)
				if params.actualTest == 1 and i == 5508 then print("Finished testing"); break; end
	end
end

function fitMasks(data)
	for i =1, #data do
		print(data[i])
		x = cv.imread{data[i]:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		y = cv.imread{data[i],cv.IMREAD_UNCHANGED} 
		x = prepare(x)
		x = x:resize(1,1,x:size(1),x:size(2)):cuda()
		pred = model:forward(x)
		predUpscaled = image.scale(pred:squeeze():double(),580,420)
		image.saveJPG(data[i]:gsub("mask","fitted"),predUpscaled)
		xlua.progress(i,#data)
		if i % 20 == 0 and params.display == 1 then
			display(x,y,predUpscaled,"train",data[i])
			sys.sleep(0.5)
		end
	end
end
if params.run == 1 then run() end
	










