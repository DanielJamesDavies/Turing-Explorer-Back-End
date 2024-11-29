// Packages

// Components

// Logic
import { TextBoxLogic } from "./TextBoxLogic";

// Context

// Services

// Styles
import "./TextBox.css";

// Assets

export const TextBox = ({ className, value, setValue, placeholder, onEnter, focusOnStart, blurOnEnter }) => {
	const { textBoxRef, textBoxHeightBoxRef, textBoxWidth, textBoxHeightBoxHeight, onTextBoxChange, onKeyDown } = TextBoxLogic({
		value,
		setValue,
		onEnter,
		focusOnStart,
		blurOnEnter,
	});

	return (
		<div className={"text-area-container" + (className ? " " + className : "")}>
			<textarea
				ref={textBoxRef}
				className='text-area'
				value={value ? value : ""}
				onChange={onTextBoxChange}
				placeholder={placeholder ? placeholder : ""}
				style={{ height: textBoxHeightBoxHeight + 4 + "px" }}
				onKeyDown={onKeyDown}
			></textarea>
			<div ref={textBoxHeightBoxRef} className='text-area-height-box' style={{ width: textBoxWidth + "px" }}>
				{value}
				{value?.split("\n")[value?.split("\n").length - 1] === "" ? <br /> : null}
			</div>
		</div>
	);
};
