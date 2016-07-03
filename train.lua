function train(inputs,target)


	local output
	local loss
	local dLoss_dO
	local batchLoss
	local targetResize

	if i == 1 then
		if model then parameters,gradParameters = model:getParameters() end
		print("Number of parameters ==>")
		print(parameters:size())
		ma = MovingAverage.new(params.ma)
		losses = {}
	end
	
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		gradParameters:zero()
		output = model:forward(inputs) -- Only one input for training unlike testing
		targetResize = image.scale(target:squeeze():double(),28,16,"bilinear"):cuda()
		loss = criterion:forward(output,targetResize)
		losses[i] = loss
		dLoss_dO = criterion:backward(output,targetResize)
		model:backward(inputs,dLoss_dO)

		return	loss, gradParameters 
	end

	_, batchLoss = optimMethod(feval,parameters,optimState)

	if i % params.ma == 0 then

		local lossesT = torch.Tensor(losses)
		MA = ma:forward(lossesT)
		print(string.format("Model %s has ma mean (%d) training loss of % f",modelName, params.ma, MA[{{-1}}]:squeeze()))
		--[[
		if i > params.ma and params.displayGraph == 1 and i % params.displayGraphFreq ==0 then 
			MA:resize(MA:size(1))
			local t = torch.range(1,MA:size(1))
			local title = string.format("Model %s has ma mean (%d) training loss of % f",modelName, params.ma, MA:mean())
			gnuplot.plot({title,t,MA})
		end
		]]--
		collectgarbage()
       	end
	if i % params.lrChange == 0 then
		local clr = params.lr
		params.lr = params.lr/params.lrDecay
		print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
	end
	if i % params.modelSave == 0 then
		print("==> Saving model " .. params.modelName .. ".")
		torch.save(params.modelName,model)
	end
	xlua.progress(i,params.nIter)
	i = i + 1
	return output,targetResize

end

