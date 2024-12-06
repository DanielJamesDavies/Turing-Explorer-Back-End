// Packages

// Components

// Logic
import { TopSequencesListLogic } from "./TopSequencesListLogic";

// Context

// Services

// Styles
import "./TopSequencesList.css";
import { LoadingCircle } from "../../../components/LoadingCircle/LoadingCircle";

// Assets

export const TopSequencesList = () => {
	const {
		latentLayer,
		latentIndex,
		latentTopSequencesCount,
		topSequencesList,
		topOtherLatents,
		isShowingAll,
		toggleIsShowingAll,
		goToLatent,
		sequenceIsShowingOtherLatents,
		setSequenceIsShowingOtherLatents,
		sequenceOtherLatentsTypes,
		sequenceOtherLatentsTypeIndex,
		decrementOtherLatentsType,
		incrementOtherLatentsType,
		isHidingCommonOtherLatents,
		toggleIsHidingCommonOtherLatents,
	} = TopSequencesListLogic();

	return (
		<div className='latent-top-sequences-list-container'>
			<div className='subtitle'>List of Latent Activated Sequences</div>
			<div className='light-text'>
				The following is a list of token sequences where the latent had the highest activation out of{" "}
				{latentTopSequencesCount?.toLocaleString()} sequences.
			</div>
			<div className='latent-top-sequences-list-header'>
				<div>Rank</div>
				<div>Average Activation</div>
				<div>Token Sequence</div>
				<div>View Top Other Latents</div>
			</div>
			<div className='latent-top-sequences-list'>
				{!topSequencesList || topSequencesList?.length === 0 ? (
					<div className='latent-top-sequences-list-loading-circle-container'>
						<LoadingCircle center={true} size='s' />
					</div>
				) : (
					topSequencesList?.slice(0, isShowingAll ? topSequencesList?.length : 5)?.map((topSequence, index) => (
						<div key={index} className='latent-top-sequences-item-container'>
							<div className='latent-top-sequences-item'>
								<div className='latent-top-sequences-item-info'>{index + 1}</div>
								<div className='latent-top-sequences-item-info'>{topSequence?.[0]?.[1]}</div>
								<div className='latent-top-sequences-item-tokens'>
									{!Array.isArray(topSequence)
										? null
										: topSequence?.map((tokenData, index2) =>
												index2 === 0 ? null : (
													<div key={index2} className='latent-top-sequences-item-token-item'>
														<div
															className='latent-top-sequences-item-token-item-value-graph'
															style={{
																"--token-value-fraction":
																	tokenData?.[1] /
																	Math.max(...topSequence?.filter((_, i) => i !== 0)?.map((e) => e[1])),
															}}
														>
															<div className='latent-top-sequences-item-token-item-value-graph-bar'></div>
														</div>
														<div className='latent-top-sequences-item-token-item-token'>{tokenData?.[0]}</div>
														<div className='latent-top-sequences-item-token-item-value'>
															{tokenData?.[1]?.toFixed(3)}
														</div>
													</div>
												)
										  )}
								</div>
								<div className='latent-top-sequences-item-info'>
									<button
										className='latent-top-sequences-item-view-top-other-latents-btn'
										onClick={
											sequenceIsShowingOtherLatents === index
												? () => setSequenceIsShowingOtherLatents(-1)
												: () => setSequenceIsShowingOtherLatents(index)
										}
									>
										{sequenceIsShowingOtherLatents === index ? (
											<i className='fa-solid fa-eye' />
										) : (
											<i className='fa-solid fa-eye-slash' />
										)}
									</button>
								</div>
							</div>
							<div
								className={
									"latent-top-sequences-item-other-latents-container" +
									(sequenceIsShowingOtherLatents === index ? " latent-top-sequences-item-other-latents-container-show" : "")
								}
							>
								<div className='latent-top-sequences-item-other-latents-collection-type-container'>
									<div className='latent-top-sequences-item-other-latents-collection-type-title'>
										Other Latents Collection Type:{" "}
									</div>
									<button
										className='latent-top-sequences-item-other-latents-collection-type-switch-btn'
										onClick={decrementOtherLatentsType}
									>
										<i className='fa-solid fa-chevron-left' />
									</button>
									<span className='latent-top-sequences-item-other-latents-collection-type-value'>
										{sequenceOtherLatentsTypes[sequenceOtherLatentsTypeIndex]?.name}
									</span>
									<button
										className='latent-top-sequences-item-other-latents-collection-type-switch-btn'
										onClick={incrementOtherLatentsType}
									>
										<i className='fa-solid fa-chevron-right' />
									</button>
									<span className='latent-top-sequences-item-other-latents-collection-type-toggle-hide-common-latents-label'>
										Hide Common Other Latents
									</span>
									<button
										className={
											"latent-top-sequences-item-other-latents-collection-type-toggle-hide-common-latents" +
											(isHidingCommonOtherLatents
												? " latent-top-sequences-item-other-latents-collection-type-toggle-hide-common-latents-active"
												: "")
										}
										onClick={toggleIsHidingCommonOtherLatents}
									></button>
								</div>
								<div className='latent-top-sequences-item-other-latents-label'>
									<i className='fa-solid fa-arrow-left-long'></i>
									<span>
										Left most latents are more connected to this latent (Layer {latentLayer + 1} Latent {latentIndex + 1})
									</span>
								</div>
								<div>
									{!topOtherLatents || topOtherLatents.length === 0 ? (
										<div className='latent-top-sequences-list-loading-circle-container'>
											<LoadingCircle center={true} size='s' />
										</div>
									) : (
										topOtherLatents[
											sequenceOtherLatentsTypes[sequenceOtherLatentsTypeIndex]?.key +
												(isHidingCommonOtherLatents ? "_rare" : "")
										]?.map((topOtherLatentsLayer, topOtherLatentsLayerIndex) => (
											<div
												key={topOtherLatentsLayerIndex}
												className={
													"latent-top-sequences-item-other-latents-layer" +
													(topOtherLatentsLayerIndex > 5
														? " latent-top-sequences-item-other-latents-layer-use-higher-preview"
														: "")
												}
												style={{ "--layer-index": topOtherLatentsLayerIndex }}
											>
												<div className='latent-top-sequences-item-other-latents-layer-title'>
													Layer {topOtherLatentsLayerIndex + 1 < 10 ? "0" : ""}
													{topOtherLatentsLayerIndex + 1}
												</div>
												<div className='latent-top-sequences-item-other-latents-layer-sequences'>
													{topOtherLatentsLayer[index]?.slice(0, 24)?.map((otherLatent, index2) => (
														<div key={index2} className='latent-top-sequences-item-other-latent-container'>
															<div
																className='inference-result-latents-latent-container'
																onMouseDown={(e) => e?.preventDefault()}
																onClick={(e) => goToLatent(e, topOtherLatentsLayerIndex, otherLatent?.latent)}
																onAuxClick={(e) => goToLatent(e, topOtherLatentsLayerIndex, otherLatent?.latent)}
															>
																<div className='inference-result-latents-latent'>
																	<button>
																		<span>{otherLatent?.latent + 1}</span>
																	</button>
																	<div className='inference-result-latents-latent-preview'>
																		<div className='inference-result-latents-latent-preview-location'>
																			<span>Layer {topOtherLatentsLayerIndex + 1}</span>
																			<span>Latent {otherLatent?.latent + 1}</span>
																		</div>
																		<div className='inference-result-latents-latent-preview-label'>
																			Top Sequences
																		</div>
																		<div className='inference-result-latents-latent-preview-top-sequences'>
																			{otherLatent?.topSequences?.map((topSequence, topSequenceIndex) => (
																				<div
																					key={topSequenceIndex}
																					className='inference-result-latents-latent-preview-top-sequence'
																				>
																					{topSequence}
																				</div>
																			))}
																		</div>
																	</div>
																</div>
															</div>
														</div>
													))}
												</div>
											</div>
										))
									)}
								</div>
							</div>
						</div>
					))
				)}
			</div>
			{!topSequencesList || topSequencesList?.length === 0 ? null : (
				<div className='latent-top-sequences-list-toggle-show-all-btn-container'>
					<button className='latent-top-sequences-list-toggle-show-all-btn' onClick={toggleIsShowingAll}>
						Show {isShowingAll ? "Less" : "More"} Latent Activated Sequences
					</button>
				</div>
			)}
		</div>
	);
};
