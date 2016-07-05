require "paths"
require "image"
require "cunn"

cv = require "cv"
require "cv.imgcodecs"

local function fileExists(name)
	local f=io.open(name,"r")
	if f~=nil then io.close(f) return true else return false end
end

loadData = {}
local trainPaths  = {}
local testPaths  = {}
for f in paths.files("train/","mask.tif") do
		trainPaths[#trainPaths+1] = "train/"..f
end
for f in paths.files("test/",".tif") do
		testPaths[#testPaths+1] = "test/"..f
end

function loadData.init(tid,nThreads)

	imgPaths = {}
	local t
	if params.actualTest == 1 then 
		t = testPaths
	else 
		t = trainPaths 
	end
	for i = tid, #t , nThreads do 
		imgPaths[#imgPaths + 1] = t[i]	
	end

	return imgPaths 
end


function loadData.loadObs(trainOrTest,imgPaths)
	local x, y
	local rObs 
	local obs
	if params.actualTest == 1 then
		if testObsIndex == nil then testObsIndex = 1 end
		if testObsIndex > #imgPaths then 
			while true do
				print("Thread sleepng ...")
				sys.sleep(15)
			end

		end
		obs = imgPaths[testObsIndex]
		x = cv.imread{obs,cv.IMREAD_UNCHANGED} 
		testObsIndex = testObsIndex + 1
		return obs, x:cuda():resize(1,1,x:size(1),x:size(2))
	end
	if trainOrTest == "train" then
		rObs = imgPaths[torch.random(#imgPaths)]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		y:div(255)

		--[[
		local randInt = torch.random(2)
		if randInt == 1 then	
		elseif randInt == 2 then
			image.hflip(x,x)
			image.hflip(y,y)
		elseif randInt == 3 then
			image.vflip(x,x)
			image.vflip(y,y)
		elseif randInt == 4 then
			image.vflip(x,x)
			image.vflip(y,y)
			image.hflip(x,x)
			image.hflip(y,y)
		end
		]]--

	elseif trainOrTest == "test" then
		if testObsIndex == nil then testObsIndex = 1 end
		rObs = imgPaths[testObsIndex]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		y:div(255)
		if testObsIndex > #rObs then
			testObsIndex = 1
		else
			testObsIndex = testObsIndex + 1
		end
	end

	return rObs, x:cuda():resize(1,1,x:size(1),x:size(2)),y:cuda():resize(y:size(1),y:size(2))
end

return loadData
