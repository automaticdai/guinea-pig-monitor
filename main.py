"""
Guinea Pig Behavior Monitoring System
Main entry point
"""
import sys
from src import GuineaPigMonitor


def print_banner():
    """Print project title and description"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     Guinea Pig Behavior Monitoring (GPM) System               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Version 1.0 • (c) 2025 automaticdai
"""
    print(banner)


def main():
    """Main entry point"""
    # Print startup banner
    print_banner()
    
    # Get RTSP URL from command line if provided
    rtsp_url = None
    if len(sys.argv) > 1:
        rtsp_url = sys.argv[1]
    
    # Create and run monitor
    monitor = GuineaPigMonitor(model_path='model/yolov11.pt', rtsp_url=rtsp_url)
    monitor.run()


if __name__ == '__main__':
    main()
