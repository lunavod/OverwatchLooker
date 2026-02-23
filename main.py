import argparse
import sys
from pathlib import Path

from overwatchlooker.config import ANALYZER
from overwatchlooker.display import print_analysis, print_error
from overwatchlooker.notification import copy_to_clipboard, show_notification
from overwatchlooker.tray import App


def main():
    parser = argparse.ArgumentParser(description="Overwatch 2 screen analyzer")
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to an image file to analyze instead of listening for hotkeys",
    )
    parser.add_argument(
        "--tg",
        action="store_true",
        help="Send results to Telegram instead of copying to clipboard",
    )
    args = parser.parse_args()

    if args.image:
        path = Path(args.image)
        if not path.is_file():
            print_error(f"File not found: {path}")
            sys.exit(1)
        png_bytes = path.read_bytes()
        if ANALYZER == "claude":
            from overwatchlooker.analyzer import analyze_screenshot
        else:
            from overwatchlooker.ocr_analyzer import analyze_screenshot
        result = analyze_screenshot(png_bytes)
        if result.startswith("NOT_OW2_TAB"):
            print_error("Image does not appear to be an OW2 Tab screen.")
        else:
            formatted = print_analysis(result)
            if args.tg:
                from overwatchlooker.telegram import send_message
                if send_message(formatted):
                    show_notification("OverwatchLooker", "Analysis sent to Telegram.")
                else:
                    print_error("Failed to send to Telegram.")
            else:
                copy_to_clipboard(formatted)
                show_notification("OverwatchLooker", "Analysis complete. Copied to clipboard.")
    else:
        app = App(use_telegram=args.tg)
        app.run()


if __name__ == "__main__":
    main()
