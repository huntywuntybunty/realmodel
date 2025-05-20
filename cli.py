#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import asyncio
import logging
from typing import Optional, Dict
from k_model import auto_project_strikeouts  # your async projection pipeline

def setup_parser() -> argparse.ArgumentParser:
    """Configure command line argument parser"""
    parser = argparse.ArgumentParser(description='MLB Strikeout Projections')
    subparsers = parser.add_subparsers(dest='command', required=True)

    predict_parser = subparsers.add_parser('predict', help='Run strikeout projections')
    predict_parser.add_argument('--pitcher', required=True, help='Pitcher full name (e.g. "Paul Skenes")')
    predict_parser.add_argument('--opponent', required=True, help='Opponent team abbreviation (e.g. "PHI")')
    predict_parser.add_argument('--park', default="", help='Ballpark name (optional)')
    predict_parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    return parser

async def run_prediction(pitcher: str, opponent: str, park: str = "") -> Optional[Dict]:
    """Wrapper function for the prediction with proper async context"""
    try:
        return await auto_project_strikeouts(pitcher, opponent, park)
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return None

def format_result(result: Dict) -> str:
    """Format the projection results for display"""
    if not result:
        return "⚠️ No projection available"
    
    output = [
        f"\n=== {result['pitcher']} vs {result['opponent']} ===",
        f"Projected Ks: {result['mean']:.1f}",
        f"Vegas Line: {result.get('vegas_line', 'N/A')}",
        f"Edge: {result.get('edge', 'N/A')}",
        f"Over 6.5 Probability: {result.get('prob_over_6.5', 0):.1f}%",
        f"Key Stats: {result.get('lineup_source', 'Lineup source unknown')}"
    ]
    return "\n".join(output)

async def main():
    """Main async entry point"""
    parser = setup_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    if args.command == 'predict':
        result = await run_prediction(args.pitcher, args.opponent, args.park)
        print(format_result(result))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        exit(1)
