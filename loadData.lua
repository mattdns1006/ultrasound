require "paths"
require "image"
require "cunn"

cv = require "cv"
local csv = require "csv"
require "cv.imgcodecs"
require "cv.imgproc"

local function fileExists(name)
	local f=io.open(name,"r")
	if f~=nil then io.close(f) return true else return false end
end

trainCsv = csv.csvToTable("train.csv")
testCsv = csv.csvToTable("test.csv") -- test csv for continous testing (not actual test)

function tableConcat(t1,t2)
	local t3 = {}
	for i=1, #t1 do
		t3[#t3+1] = t1[i]
	end
	for i=1, #t2 do
		t3[#t3+1] = t2[i]
	end
	return t3
end
bothCsvs = tableConcat(trainCsv,testCsv)

loadData = {}
local trainPaths  = {}
local testPaths  = {}
for f in paths.files("train/","mask") do
		trainPaths[#trainPaths+1] = "train/" .. f
end
for f in paths.files("test/",".tif") do -- actual test
		testPaths[#testPaths+1] = f
end

function loadData.init(tid,nThreads)

	local imgPaths = {}
	local t
	if params.actualTest == 1 then 
		print("Thread ==>", tid, " actual testing.")
		t = testPaths
		for i = tid, #t , nThreads do 
			imgPaths[#imgPaths + 1] = t[i]	
		end
	elseif params.trainAll == 1 then 
		t = bothCsvs
		for i = tid, #t , nThreads do 
			imgPaths[#imgPaths + 1] = t[i]	
		end
		print("Thread ==>", tid, " training on everything. Number of observations = ", #imgPaths)

	elseif tid == 1 then 
		imgPaths = testCsv --for thread 1 for continuous testing
		print("Thread ==>", tid, " testing on subset. Number of observations = ", #imgPaths)
	else 
		t = trainCsv

		for i = tid, #t , nThreads -1 do 
			imgPaths[#imgPaths + 1] = t[i]	
		end
		print("Thread ==>", tid, " training on subset. Number of observations = ", #imgPaths)
	end

	return imgPaths 
end

function prepare(img)
	local imgOut
	imgOut = image.scale(img:squeeze(),params.inSize[4],params.inSize[3],"bilinear"):double()
	imgOut:csub(imgOut:mean()) -- remove mean for brightness
	return imgOut
end

function augment(x,y)
	local h,w = x:size(1), x:size(2)
	local center = cv.Point2f{w/2,h/2}
	local angle = torch.uniform(-5,5)
	local scale = 1 + torch.rand(1)[1]*0.01
	local tx ,ty = torch.random(-15,15), torch.random(-15,15)
	local M = cv.getRotationMatrix2D{center,angle,scale}
	M[1][3] = tx
	M[2][3] = ty
	local x1 = cv.warpAffine{x,M,flags=cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE}
	local y1 = cv.warpAffine{y,M,flags=cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE}
	return x1,y1
end

function augmentExample()
	local initPic = torch.range(1,torch.pow(100,2),1):reshape(100,100)
	zoom = 4
	imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
	imgDisplay1 = image.display{image=initPic, zoom=zoom, offscreen=false}
	for j=1,200 do
		rObs = trainCsv[torch.random(#trainCsv)]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 

		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		for i=1, 4 do
			dstX, dstY = augment(x,y)
			image.display{image = dstX, win = imgDisplay0, legend = " x"}
			image.display{image = dstY, win = imgDisplay1, legend =  " y"}
			sys.sleep(0.2)
		end
	end
end

function loadData.loadObs(trainOrTest,imgPaths)
	local x, y
	local rObs 
	local obs
	if obsIdx == nil then obsIdx = 1 end

	if params.actualTest == 1 then
		if obsIdx > #imgPaths then 
			while true do
				print("Thread sleeping ...")
				sys.sleep(15)
			end

		end
		obs = imgPaths[obsIdx]
		x = cv.imread{"test/"..obs,cv.IMREAD_UNCHANGED} 
		obsIdx = obsIdx + 1
		x = prepare(x)
		return obs, x:cuda():resize(1,1,x:size(1),x:size(2))
	end
	if trainOrTest == "train" then
		rObs = imgPaths[obsIdx]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		y:div(255)
		x,y = augment(x,y)

	elseif trainOrTest == "test" then
		rObs = imgPaths[obsIdx]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		y:div(255)

	end
	x = prepare(x)
	if obsIdx == #imgPaths then
		obsIdx = 1
	else
		obsIdx = obsIdx + 1
	end

	return rObs, x:cuda():resize(1,1,x:size(1),x:size(2)),y:cuda():resize(y:size(1),y:size(2))
end

return loadData
