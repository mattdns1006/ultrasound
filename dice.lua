function diceCoeff(pred,truth,threshold)
	local smooth = 1
	if diceThreshold == nil then diceThreshold = 0.5 end
	pred[pred:le(threshold)] = 0
	pred[pred:gt(threshold)] = 1
	local intersection = torch.cmul(pred,truth):sum() 
	return (intersection*2 + smooth)/(pred:sum() + truth:sum() + smooth)
end

