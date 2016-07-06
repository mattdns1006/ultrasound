function diceCoeff(pred,truth,diceThreshold)
	local smooth = 1
	if diceThreshold == nil then diceThreshold = 0.5 end
	pred1 = torch.Tensor(pred:size()):copy(pred):cuda()
	pred1[pred1:le(diceThreshold)] = 0
	pred1[pred1:gt(diceThreshold)] = 1
	local intersection = torch.cmul(pred1,truth):sum() 
	return (intersection*2 + smooth)/(pred1:sum() + truth:sum() + smooth)
end

function diceROC(pred,truth)
	local thresholds = torch.linspace(0.05,0.95,25)
	local diceCoeffs = {}
	for i =1,thresholds:size(1) do 
		diceCoeffs[#diceCoeffs+1] = diceCoeff(pred,truth,thresholds[i])
	end
	return diceCoeffs
end

function egDice()
	require "cutorch"
	require "cunn"
	dice = {}
	for i =1, 100 do
		x = torch.rand(10,10):cuda()
		y = torch.zeros(10,10):cuda()
		y:narrow(1,2,4):narrow(2,3,4):fill(1)
		dice[#dice +1]=diceROC(x,y)
	end
	return dice
end
