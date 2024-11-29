import { Route, Routes } from "react-router-dom";

import { NavigationBar } from "./components/NavigationBar/NavigationBar";

import { SearchPage } from "./pages/SearchPage/SearchPage";
import { LatentPage } from "./pages/LatentPage/LatentPage";
import { InferencePage } from "./pages/InferencePage/InferencePage";

function App() {
	return (
		<div className='app'>
			<NavigationBar />
			<Routes>
				<Route>
					<Route path='/' element={<SearchPage />} />
					<Route path='/latent' element={<LatentPage />} />
					<Route path='/inference' element={<InferencePage />} />
				</Route>
			</Routes>
		</div>
	);
}

export default App;
