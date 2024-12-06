import React, { createContext, useCallback } from "react";

export const APIContext = createContext();

const APIProvider = ({ children }) => {
	const APIRequest = useCallback(async (path, method, body) => {
		let data = {
			method,
			headers: {
				"Content-Type": "application/json",
			},
		};

		if (body) data.body = JSON.stringify(body);

		try {
			const API_URL = `http://${window.location?.hostname}:5000/api`;

			const response = await fetch(API_URL + path, data);

			const responseData = await response.json();

			return responseData;
		} catch (e) {
			return { errors: [{ message: "Failed to Send Request to Server" }] };
		}
	}, []);

	return <APIContext.Provider value={{ APIRequest }}>{children}</APIContext.Provider>;
};

export default APIProvider;
