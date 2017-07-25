function  PcdfRand()
	mix = 2;
	w = zeros(mix,1);
	w(1) = 0.4;
	w(2) = 0.6;
	Lambda = zeros(mix,1);
	Lambda(1) = 4;
	Lambda(2) = 20;
	maxX = 100;
	% PDF
	proB =zeros(maxX,1);
	for Index = 1:maxX
		proB(Index) = w(1)*Lambda(1)^Index*exp(-Lambda(1))/prod(1:Index)+w(2)*Lambda(2)^Index*exp(-Lambda(2))/prod(1:Index);
	end

	%CDF
	cdfProb = cumsum(proB);
    figure;
    plot(proB);
	figure;
	plot(cdfProb);



end