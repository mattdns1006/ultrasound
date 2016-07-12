Threads = require "threads"

do 
	local threadParams = params
	donkeys = Threads(
			params.nThreads,
			function(idx)
				params = threadParams
				require "torch"
				--require "cutorch"
				require "xlua"
				require "string"
				require "image"
				--cutorch.setDevice(1)

				tid = idx -- Thread id
				print(string.format("Initialized thread %d of %d.", tid,params.nThreads))
				loadData = require "loadData"
				imgPaths = loadData.init(tid,params.nThreads)
			end
			)
end
	
