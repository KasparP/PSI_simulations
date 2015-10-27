function x = prox_maxk(v,k)
	N = length(v(:));
	idxkeep = v > prctile(v(:),(N-k)/N*100);
	x = v.*idxkeep;
end