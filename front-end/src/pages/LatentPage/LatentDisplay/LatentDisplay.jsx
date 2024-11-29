// Packages

// Components

// Logic
import { LatentDisplayLogic } from "./LatentDisplayLogic";

// Context

// Services

// Styles
import "./LatentDisplay.css";

// Assets

export const LatentDisplay = () => {
	const { latentFrequencyTokensCount, latentLayer, latentIndex, latentFrequency } = LatentDisplayLogic();

	return (
		<div className='latent-display'>
			<div className='latent-display-latent-info-container'>
				<div className='latent-display-latent-info-above-latent'>
					<div className='latent-display-latent-info'>
						<div>Layer</div>
						<div className='latent-display-latent-info-value'>{latentLayer + 1}</div>
						<div className='latent-display-latent-info-label'></div>
					</div>
					<div className='latent-display-latent-info'>
						<div>Latent</div>
						<div className='latent-display-latent-info-value'>{latentIndex + 1}</div>
						<div className='latent-display-latent-info-label'></div>
					</div>
				</div>
				<div className='latent-display-latent-info-below-latent'>
					<div className='latent-display-latent-info'>
						<div>Activated</div>
						<div className='latent-display-latent-info-value'>
							{latentFrequency.toLocaleString()}{" "}
							{isNaN(latentFrequency / latentFrequencyTokensCount)
								? ""
								: "(" + ((latentFrequency / latentFrequencyTokensCount) * 100).toFixed(2) + "%)"}
						</div>
						<div>Times Over {latentFrequencyTokensCount.toLocaleString()} Sequences</div>
					</div>
				</div>
			</div>
			<div className='latent-display-latent-container'>
				<div className='latent-circle latent-circle-l'></div>
			</div>
			<div className='latent-display-latent-weights-container'>
				{Array(2)
					.fill(0)
					.map((_, index) => (
						<div key={index} className='latent-display-latent-weights'>
							<div className='latent-display-latent-weights-left'>
								{Array(9)
									.fill(0)
									.map((_, index) => (
										<div key={index} className='latent-display-latent-weight-line'></div>
									))}
							</div>
							<div className='latent-display-latent-weights-right'>
								{Array(9)
									.fill(0)
									.map((_, index) => (
										<div key={index} className='latent-display-latent-weight-line'></div>
									))}
							</div>
						</div>
					))}
			</div>
		</div>
	);
};
