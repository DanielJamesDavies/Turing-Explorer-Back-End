import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import "./index.css";

import APIProvider from "./context/APIContext.jsx";
import SearchProvider from "./context/SearchContext.jsx";
import InferenceProvider from "./context/InferenceContext.jsx";
import LatentProvider from "./context/LatentContext.jsx";

createRoot(document.getElementById("root")).render(
	<StrictMode>
		<BrowserRouter>
			<APIProvider>
				<SearchProvider>
					<InferenceProvider>
						<LatentProvider>
							<App />
						</LatentProvider>
					</InferenceProvider>
				</SearchProvider>
			</APIProvider>
		</BrowserRouter>
	</StrictMode>
);
