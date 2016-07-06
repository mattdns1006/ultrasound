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


local trainCsv = csv.csvToTable("train.csv")
local testCsv = csv.csvToTable("test.csv") -- test csv for continous testing (not actual test)

loadData = {}
local trainPaths  = {}
testPaths  = {}
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
	elseif params.fullTrain == 1 then 
		print("Thread ==>", tid, " training on everything.")
		t = trainCsv
		for i = tid, #t , nThreads do 
			imgPaths[#imgPaths + 1] = t[i]	
		end
	elseif tid == 1 then 
		print("Thread ==>", tid, " testing on subset.")
		imgPaths = testCsv --for thread 1 for continuous testing
	else 
		print("Thread ==>", tid, " training on subset.")
		t = trainCsv
		for i = tid, #t , nThreads -1 do 
			imgPaths[#imgPaths + 1] = t[i]	
		end
	end

	return imgPaths 
end

local function prepare(img)
	local imgOut
	if params.level ~= 0 then
		imgOut = image.scale(img:squeeze(),params.inSize[4],params.inSize[3],"bilinear")
		imgOut:csub(imgOut:mean()) -- remove mean for brightness
	end
	return imgOut
end

local function augment(x,y)
	if torch.uniform() < 0.8 then
		h,w = x:size(1), x:size(2)
		center = cv.Point2f{w/2,h/2}
		angle = torch.uniform(-5,5)
		scale = 1 + torch.rand(1)[1]*0.03
		M = cv.getRotationMatrix2D{center,angle,scale}
		x = cv.warpAffine{x,M,flags=cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE}
		y = cv.warpAffine{y,M,flags=cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT}
	end
	return x,y
end

function augmentExample()
	local initPic = torch.range(1,torch.pow(100,2),1):reshape(100,100)
	zoom = 4
	imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
	imgDisplay1 = image.display{image=initPic, zoom=zoom, offscreen=false}
	for j=1,200 do
		rObs = trainCsv[torch.random(#trainCsv)]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		x = x:narrow(1,2,x:size(1)-2):narrow(2,2,x:size(2)-2)
		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		for i=1, 5 do
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

	if params.actualTest == 1 then
		if testObsIndex == nil then testObsIndex = 1 end
		if testObsIndex > #imgPaths then 
			while true do
				print("Thread sleeping ...")
				sys.sleep(15)
			end

		end
		obs = imgPaths[testObsIndex]
		x = cv.imread{"test/"..obs,cv.IMREAD_UNCHANGED} 
		testObsIndex = testObsIndex + 1
		x = prepare(x)
		return obs, x:cuda():resize(1,1,x:size(1),x:size(2))
	end
	if trainOrTest == "train" then
		rObs = imgPaths[torch.random(#imgPaths)]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		y:div(255)
		x,y = augment(x,y)

	elseif trainOrTest == "test" then
		if testObsIndex == nil then testObsIndex = 1 end
		rObs = imgPaths[testObsIndex]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		y:div(255)
		if testObsIndex == #imgPaths then
			testObsIndex = 1
		else
			testObsIndex = testObsIndex + 1
		end
	end
	x = prepare(x)

	return rObs, x:cuda():resize(1,1,x:size(1),x:size(2)),y:cuda():resize(y:size(1),y:size(2))
end

return loadData
