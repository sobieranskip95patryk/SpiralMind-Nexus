"""Command Line Interface for SpiralMind-Nexus.

Provides CLI access to all SpiralMind-Nexus functionality.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from .pipeline.double_pipeline import execute, batch_execute, get_pipeline_statistics, reset_pipeline
from .config.loader import load_config
from .utils.logging_config import setup_logging, get_logger
from .memory.persistence import MemoryPersistence
from . import __version__

# Setup logging
logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog='spiral',
        description='SpiralMind-Nexus - Advanced Text Processing and Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  spiral --text "Hello world" --mode quantum
  spiral --batch input.txt --output results.json
  spiral --text "Complex analysis" --save-memory --verbose
  spiral --statistics
  spiral --config config.yaml --text "Test" --mode gokai
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'SpiralMind-Nexus {__version__}'
    )
    
    # Text input options
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        '--text', '-t',
        type=str,
        help='Text to process'
    )
    text_group.add_argument(
        '--batch', '-b',
        type=Path,
        help='File containing texts to process (one per line)'
    )
    text_group.add_argument(
        '--file', '-f',
        type=Path,
        help='Single file to process'
    )
    
    # Processing options
    parser.add_argument(
        '--mode', '-m',
        choices=['quantum', 'gokai', 'hybrid', 'debug'],
        default='quantum',
        help='Processing mode (default: quantum)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing for batch mode'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Configuration file path'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file for results (JSON format)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'pretty', 'csv', 'minimal'],
        default='pretty',
        help='Output format (default: pretty)'
    )
    
    # Memory options
    parser.add_argument(
        '--save-memory',
        action='store_true',
        help='Save results to memory persistence'
    )
    
    parser.add_argument(
        '--memory-search',
        type=str,
        help='Search memory for text patterns'
    )
    
    parser.add_argument(
        '--memory-export',
        type=Path,
        help='Export memory to JSON file'
    )
    
    # Statistics and management
    parser.add_argument(
        '--statistics', '--stats',
        action='store_true',
        help='Show pipeline statistics'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset pipeline statistics and cache'
    )
    
    parser.add_argument(
        '--cleanup-memory',
        type=int,
        metavar='DAYS',
        help='Clean up memories older than N days'
    )
    
    # Context options
    parser.add_argument(
        '--context',
        type=str,
        help='JSON string with processing context'
    )
    
    parser.add_argument(
        '--importance',
        type=float,
        metavar='0.0-1.0',
        help='Importance score for processing context'
    )
    
    parser.add_argument(
        '--urgency',
        type=float,
        metavar='0.0-1.0',
        help='Urgency score for processing context'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (-v for DEBUG, -vv for extra debug)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Log to file'
    )
    
    return parser


def setup_cli_logging(args: argparse.Namespace) -> None:
    """Setup logging based on CLI arguments.
    
    Args:
        args: Parsed command line arguments
    """
    if args.quiet:
        level = "ERROR"
    elif args.verbose >= 2:
        level = "DEBUG"
    elif args.verbose == 1:
        level = "INFO"
    else:
        level = "WARNING"
    
    setup_logging(
        level=level,
        log_file=str(args.log_file) if args.log_file else None,
        enable_console=not args.quiet
    )


def load_texts_from_file(file_path: Path) -> List[str]:
    """Load texts from file.
    
    Args:
        file_path: Path to input file
        
    Returns:
        List of text strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    return [str(item) for item in data]
                else:
                    return [str(data)]
            else:
                # Treat as text file with one text per line
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        sys.exit(1)


def build_context(args: argparse.Namespace) -> Dict[str, Any]:
    """Build processing context from CLI arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Context dictionary
    """
    context = {
        'cli_mode': True,
        'timestamp': datetime.now().isoformat()
    }
    
    if args.context:
        try:
            user_context = json.loads(args.context)
            context.update(user_context)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in context: {e}")
            sys.exit(1)
    
    if args.importance is not None:
        if not 0.0 <= args.importance <= 1.0:
            logger.error("Importance must be between 0.0 and 1.0")
            sys.exit(1)
        context['importance'] = args.importance
    
    if args.urgency is not None:
        if not 0.0 <= args.urgency <= 1.0:
            logger.error("Urgency must be between 0.0 and 1.0")
            sys.exit(1)
        context['urgency'] = args.urgency
    
    return context


def format_output(results: Any, format_type: str) -> str:
    """Format output based on requested format.
    
    Args:
        results: Results to format
        format_type: Output format type
        
    Returns:
        Formatted string
    """
    if format_type == 'json':
        return json.dumps(results, indent=2, default=str)
    
    elif format_type == 'minimal':
        if isinstance(results, list):
            outputs = []
            for i, result in enumerate(results):
                if result.get('success'):
                    score = result.get('quantum_score', 0.0)
                    decision = result.get('decision', 'unknown')
                    outputs.append(f"{i}: {score:.3f} ({decision})")
                else:
                    outputs.append(f"{i}: ERROR - {result.get('error', 'Unknown')}")
            return '\n'.join(outputs)
        else:
            if results.get('success'):
                score = results.get('quantum_score', 0.0)
                decision = results.get('decision', 'unknown')
                return f"{score:.3f} ({decision})"
            else:
                return f"ERROR: {results.get('error', 'Unknown')}"
    
    elif format_type == 'csv':
        if isinstance(results, list):
            lines = ['index,quantum_score,gokai_score,decision,confidence,success,error']
            for i, result in enumerate(results):
                q_score = result.get('quantum_score', 0.0)
                g_score = result.get('gokai_score', 0.0)
                decision = result.get('decision', 'unknown')
                confidence = result.get('confidence', 0.0)
                success = result.get('success', False)
                error = result.get('error', '').replace(',', ';')  # Escape commas
                lines.append(f"{i},{q_score:.3f},{g_score:.3f},{decision},{confidence:.3f},{success},{error}")
            return '\n'.join(lines)
        else:
            q_score = results.get('quantum_score', 0.0)
            g_score = results.get('gokai_score', 0.0)
            decision = results.get('decision', 'unknown')
            confidence = results.get('confidence', 0.0)
            success = results.get('success', False)
            error = results.get('error', '').replace(',', ';')
            return f"quantum_score,gokai_score,decision,confidence,success,error\n{q_score:.3f},{g_score:.3f},{decision},{confidence:.3f},{success},{error}"
    
    else:  # pretty format
        if isinstance(results, list):
            output_lines = []
            for i, result in enumerate(results):
                output_lines.append(f"\n--- Result {i+1} ---")
                if result.get('success'):
                    output_lines.append(f"Quantum Score: {result.get('quantum_score', 0.0):.3f}")
                    output_lines.append(f"GOKAI Score:   {result.get('gokai_score', 0.0):.3f}")
                    output_lines.append(f"Decision:      {result.get('decision', 'unknown')}")
                    output_lines.append(f"Confidence:    {result.get('confidence', 0.0):.3f}")
                    output_lines.append(f"Processing:    {result.get('processing_time', 0.0):.3f}s")
                else:
                    output_lines.append(f"ERROR: {result.get('error', 'Unknown error')}")
            return '\n'.join(output_lines)
        else:
            output_lines = ["\n=== SpiralMind-Nexus Results ==="]
            if results.get('success'):
                output_lines.append(f"Quantum Score: {results.get('quantum_score', 0.0):.3f}")
                output_lines.append(f"GOKAI Score:   {results.get('gokai_score', 0.0):.3f}")
                output_lines.append(f"Decision:      {results.get('decision', 'unknown')}")
                output_lines.append(f"Confidence:    {results.get('confidence', 0.0):.3f}")
                output_lines.append(f"Processing:    {results.get('processing_time', 0.0):.3f}s")
            else:
                output_lines.append(f"ERROR: {results.get('error', 'Unknown error')}")
            return '\n'.join(output_lines)


def handle_memory_operations(args: argparse.Namespace) -> bool:
    """Handle memory-related operations.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if memory operation was handled, False otherwise
    """
    memory = MemoryPersistence()
    
    if args.memory_search:
        logger.info(f"Searching memory for: {args.memory_search}")
        results = memory.search_memories(args.memory_search, limit=50)
        
        if results:
            print(f"\nFound {len(results)} memories:")
            for result in results:
                print(f"\n--- Memory ID: {result['id']} ---")
                print(f"Type: {result['memory_type']}")
                print(f"Timestamp: {result['timestamp']}")
                print(f"Data: {json.dumps(result['data'], indent=2)[:200]}...")
        else:
            print("No memories found matching the search.")
        
        return True
    
    if args.memory_export:
        logger.info(f"Exporting memory to: {args.memory_export}")
        count = memory.export_memories(str(args.memory_export))
        print(f"Exported {count} memories to {args.memory_export}")
        return True
    
    if args.cleanup_memory is not None:
        logger.info(f"Cleaning up memories older than {args.cleanup_memory} days")
        count = memory.cleanup_old_memories(args.cleanup_memory)
        print(f"Cleaned up {count} old memories")
        return True
    
    return False


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_cli_logging(args)
    
    try:
        # Handle statistics
        if args.statistics:
            stats_result = get_pipeline_statistics()
            if stats_result['success']:
                print(format_output(stats_result['statistics'], args.format))
            else:
                logger.error(f"Error getting statistics: {stats_result['error']}")
                sys.exit(1)
            return
        
        # Handle reset
        if args.reset:
            reset_result = reset_pipeline()
            if reset_result['success']:
                print(reset_result['message'])
            else:
                logger.error(f"Error resetting pipeline: {reset_result['error']}")
                sys.exit(1)
            return
        
        # Handle memory operations
        if handle_memory_operations(args):
            return
        
        # Load configuration
        if args.config:
            try:
                load_config(str(args.config))
                logger.info(f"Loaded configuration from {args.config}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                sys.exit(1)
        
        # Build processing context
        context = build_context(args)
        
        # Process text input
        results = None
        
        if args.text:
            # Single text processing
            logger.info("Processing single text")
            results = execute(
                text=args.text,
                context=context,
                mode=args.mode,
                save_to_memory=args.save_memory
            )
            
        elif args.file:
            # Single file processing
            logger.info(f"Processing file: {args.file}")
            texts = load_texts_from_file(args.file)
            if len(texts) == 1:
                results = execute(
                    text=texts[0],
                    context=context,
                    mode=args.mode,
                    save_to_memory=args.save_memory
                )
            else:
                results = batch_execute(
                    texts=texts,
                    contexts=[context] * len(texts),
                    mode=args.mode,
                    parallel=args.parallel,
                    save_to_memory=args.save_memory
                )
                
        elif args.batch:
            # Batch processing
            logger.info(f"Processing batch file: {args.batch}")
            texts = load_texts_from_file(args.batch)
            results = batch_execute(
                texts=texts,
                contexts=[context] * len(texts),
                mode=args.mode,
                parallel=args.parallel,
                save_to_memory=args.save_memory
            )
            
        else:
            # Interactive mode - read from stdin
            logger.info("Reading from stdin (enter empty line to process)")
            lines = []
            try:
                for line in sys.stdin:
                    line = line.strip()
                    if line:
                        lines.append(line)
                    else:
                        break
                
                if lines:
                    text = '\n'.join(lines)
                    results = execute(
                        text=text,
                        context=context,
                        mode=args.mode,
                        save_to_memory=args.save_memory
                    )
                else:
                    logger.error("No input provided")
                    sys.exit(1)
                    
            except KeyboardInterrupt:
                logger.info("\nOperation cancelled")
                sys.exit(0)
        
        # Output results
        if results:
            output = format_output(results, args.format)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    if args.format == 'json':
                        json.dump(results, f, indent=2, default=str)
                    else:
                        f.write(output)
                logger.info(f"Results saved to {args.output}")
            else:
                print(output)
        else:
            logger.error("No results to display")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
