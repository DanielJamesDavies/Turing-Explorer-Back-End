import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { APIContext } from "./APIContext";

export const SearchContext = createContext();

const SearchProvider = ({ children }) => {
	const { APIRequest } = useContext(APIContext);
	const [searchBarValue, setSearchBarValue] = useState("");
	const [isGettingSearchResults, setIsGettingSearchResults] = useState(false);
	const [searchResults, setSearchResults] = useState(false);
	const location = useLocation();

	const changeSearchBarValue = useCallback(
		(e) => {
			if (!isGettingSearchResults) setSearchBarValue(e?.target?.value);
		},
		[setSearchBarValue, isGettingSearchResults]
	);

	const getSearchResults = useCallback(async () => {
		const newSearchBarValue = JSON.parse(JSON.stringify(searchBarValue));
		setSearchResults(false);
		setIsGettingSearchResults(true);
		const res = await APIRequest("/search?q=" + encodeURI(newSearchBarValue));
		if (res?.results) setSearchResults({ query: newSearchBarValue, latents: res?.results });
		setIsGettingSearchResults(false);
	}, [searchBarValue, setSearchResults, setIsGettingSearchResults]);

	useEffect(() => {
		const isOnHomePage = location?.pathname?.replace("/", "")?.trim()?.length === 0;
		if (!isOnHomePage) {
			setSearchResults(false);
		}
	}, [location]);

	return (
		<SearchContext.Provider
			value={{
				searchBarValue,
				setSearchBarValue,
				changeSearchBarValue,
				isGettingSearchResults,
				setIsGettingSearchResults,
				searchResults,
				setSearchResults,
				getSearchResults,
			}}
		>
			{children}
		</SearchContext.Provider>
	);
};

export default SearchProvider;
