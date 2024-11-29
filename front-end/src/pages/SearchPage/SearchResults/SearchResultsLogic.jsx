// Packages
import { useContext } from "react";
import { useNavigate } from "react-router-dom";

// Components

// Logic

// Context
import { SearchContext } from "../../../context/SearchContext";
import { LatentContext } from "../../../context/LatentContext";

// Services

// Styles

// Assets

export const SearchResultsLogic = () => {
	const { setSearchBarValue, searchResults, isGettingSearchResults } = useContext(SearchContext);
	const { setLatentLayer, setLatentIndex } = useContext(LatentContext);
	const navigate = useNavigate();

	const goToLatent = (e, newLayer, newLatent) => {
		if (e?.button === 1)
			return window.open(window?.location?.origin + "/latent?layer=" + (newLayer + 1) + "&latent=" + (newLatent + 1), "_blank");
		setSearchBarValue("");
		setLatentLayer(newLayer);
		setLatentIndex(newLatent);
		navigate("/latent?layer=" + (newLayer + 1) + "&latent=" + (newLatent + 1));
	};

	return { isGettingSearchResults, searchResults, goToLatent };
};
