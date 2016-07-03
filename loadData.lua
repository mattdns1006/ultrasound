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

loadData.trainPaths = {}
loadData.testPaths = {}

local path = "train/"
for f in paths.files(path,"mask.tif") do
	loadData.trainPaths[#loadData.trainPaths+1] = path..f
end

function loadData.loadObs(trainOrTest)
	local x, y
	local rObs 
	if trainOrTest == "train" then
		rObs = loadData.trainPaths[torch.random(#loadData.trainPaths)]
		x = cv.imread{rObs:gsub("_mask",""),cv.IMREAD_UNCHANGED} 
		y = cv.imread{rObs,cv.IMREAD_UNCHANGED} 
		y:div(255-0)

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
	end

	-- Random flips
	return x:cuda():resize(1,1,x:size(1),x:size(2)),y:cuda():resize(y:size(1),y:size(2))
end

return loadData
