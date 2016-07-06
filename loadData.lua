require "paths"
require "image"
require "cunn"

cv = require "cv"
local csv = require "csv"
require "cv.imgcodecs"

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


function loadData.loadObs(trainOrTest,imgPaths)
	local x, y
	local rObs 
	local obs
	local function rescale(img)
		local imgOut
		if params.level ~= 0 then
			imgOut = image.scale(img:squeeze(),params.inSize[4],params.inSize[3],"bilinear")
		end
		return imgOut
	end
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
		x = rescale(x)
		return obs, x:cuda():resize(1,1,x:size(1),x:size(2))
	end
	if trainOrTest == "train" then
		rObs = imgPaths[torch.random(#imgPaths)]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		y:div(255)

		local randInt = torch.random(2)
		if randInt == 1 then	
			-- nothing
		else
			image.hflip(x,x)
			image.hflip(y,y)
		end

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
	x = rescale(x)

	return rObs, x:cuda():resize(1,1,x:size(1),x:size(2)),y:cuda():resize(y:size(1),y:size(2))
end

return loadData
