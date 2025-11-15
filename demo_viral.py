#!/usr/bin/env python3
"""Viral Demo Script for SpiralMind-Nexus.

This script demonstrates the power and capabilities of the SpiralMind-Nexus
text processing system with engaging examples and real-world use cases.
"""

import time
from datetime import datetime
import random

# Import SpiralMind-Nexus components
from spiral import execute, batch_execute, get_logger
from spiral.core import QuantumCore, GOKAICalculator
from spiral.memory import MemoryPersistence

# Setup logging
logger = get_logger(__name__)

# Demo texts showcasing different types of content
DEMO_TEXTS = {
    "viral_tweet": "🚀 Mind = BLOWN! Just discovered this AI can analyze the mathematical DNA of text using Fibonacci sequences and quantum entropy! The future is NOW! #AI #Innovation #TechRevolution",
    
    "scientific_paper": "The implementation of quantum-inspired algorithms for natural language processing represents a paradigm shift in computational linguistics. By leveraging Fibonacci sequence alignment and Shannon entropy calculations, we demonstrate significant improvements in text complexity analysis and semantic understanding.",
    
    "marketing_copy": "Transform your business with revolutionary text analysis! Our cutting-edge quantum processing technology delivers unprecedented insights into content quality, engagement potential, and viral probability. Don't just create content—CREATE IMPACT!",
    
    "philosophical_text": "In the infinite dance of words and meaning, we find that language itself follows mathematical patterns as old as nature. The spiral of human communication mirrors the golden ratio found in nautilus shells and galaxy formations.",
    
    "technical_documentation": "Initialize the quantum processing pipeline by instantiating the QuantumCore class with appropriate configuration parameters. Set weights for fibonacci (0.3), entropy (0.25), complexity (0.25), and s9 formula (0.2) calculations.",
    
    "creative_writing": "The quantum whispers of digital consciousness awakened in the silicon dreams of artificial minds, where Fibonacci spirals danced with Shannon's entropy in an eternal ballet of information and meaning.",
    
    "news_headline": "Breaking: Revolutionary AI System Discovers Hidden Mathematical Patterns in Viral Content - Scientists Amazed by Quantum Text Analysis Breakthrough!",
    
    "user_review": "OMG this is incredible! I never thought math could be this cool. The way it analyzes text is like magic but with REAL SCIENCE behind it. 5 stars! ⭐⭐⭐⭐⭐",
    
    "simple_text": "Hello world",
    
    "complex_analysis": "The quantum-mechanical interpretation of textual analysis through the lens of information theory suggests that natural language processing can benefit significantly from mathematical frameworks originally developed for quantum systems. The superposition of semantic states, entanglement of contextual meanings, and the measurement problem in natural language understanding create a rich tapestry of computational challenges that mirror fundamental questions in quantum physics."
}

# Viral content examples with known high-engagement patterns
VIRAL_EXAMPLES = [
    "🔥 THREAD: How I used quantum text analysis to predict viral content (and made $100k) 🧵👇",
    "This AI just scored my tweet and predicted it would go viral. It was RIGHT. Mind = blown 🤯",
    "POV: You discover that successful content follows mathematical patterns from nature 🌊📊✨",
    "Scientists HATE this one simple trick that predicts viral content using quantum physics! 😱🧬",
    "Breaking: Your writing style has a mathematical fingerprint (and it's beautiful) 🧮💎"
]

def print_banner():
    """Print an eye-catching banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🌀 SPIRALMIND-NEXUS DEMO 🌀                        ║
║                    Quantum-Inspired Text Analysis System                     ║
║                         🚀 PREPARE TO BE AMAZED! 🚀                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("\n🎯 Welcome to the future of text analysis!")
    print("📊 Watch as we decode the mathematical DNA of human language...\n")

def demonstrate_quantum_processing():
    """Demonstrate quantum processing capabilities."""
    print("\n" + "="*80)
    print("🔬 QUANTUM PROCESSING DEMONSTRATION")
    print("="*80)
    
    quantum = QuantumCore()
    
    for name, text in list(DEMO_TEXTS.items())[:3]:
        print(f"\n📝 Analyzing: {name.replace('_', ' ').title()}")
        print(f"💬 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        result = quantum.process(text)
        
        print("\n🧮 QUANTUM ANALYSIS RESULTS:")
        print(f"   🌀 Fibonacci Score:  {result.fibonacci_score:.3f}")
        print(f"   📊 Entropy Score:    {result.entropy_score:.3f}")
        print(f"   🧠 Complexity Score: {result.complexity_score:.3f}")
        print(f"   ⚡ S9 Formula Score: {result.s9_score:.3f}")
        print(f"   🎯 QUANTUM SCORE:    {result.quantum_score:.3f} ⭐")
        
        # Add interpretation
        if result.quantum_score > 0.7:
            print("   🔥 VERDICT: HIGH-QUALITY CONTENT! This text has strong mathematical harmony!")
        elif result.quantum_score > 0.5:
            print("   ✅ VERDICT: Good content with balanced complexity and structure.")
        else:
            print("   💡 VERDICT: Room for improvement in complexity and entropy balance.")
        
        time.sleep(1.5)  # Dramatic pause

def demonstrate_gokai_scoring():
    """Demonstrate GOKAI scoring system."""
    print("\n" + "="*80)
    print("🎯 GOKAI SCORING SYSTEM DEMONSTRATION")
    print("="*80)
    
    gokai = GOKAICalculator()
    
    for i, text in enumerate(VIRAL_EXAMPLES):
        print(f"\n📱 Analyzing Viral Example #{i+1}:")
        print(f"💬 {text}")
        
        # Create context that simulates social media metrics
        context = {
            'importance': random.uniform(0.6, 0.95),
            'urgency': random.uniform(0.5, 0.9),
            'quality': random.uniform(0.7, 0.95),
            'timestamp': datetime.now().isoformat(),
            'platform': 'social_media',
            'content_type': 'viral_candidate'
        }
        
        # First get quantum score
        quantum = QuantumCore()
        quantum_result = quantum.process(text)
        
        # Then calculate GOKAI score
        gokai_score = gokai.calculate(quantum_result.quantum_score, text, context)
        
        print("\n🎯 GOKAI ANALYSIS:")
        print(f"   💎 GOKAI Score:      {gokai_score.value:.3f}")
        print(f"   🎪 Confidence:       {gokai_score.confidence:.3f}")
        print(f"   ⚡ Weighted Score:    {gokai_score.weighted_score:.3f}")
        
        # Component breakdown
        if gokai_score.components:
            print("   📊 Component Breakdown:")
            for component, score in gokai_score.components.items():
                print(f"      {component.title():>12}: {score:.3f}")
        
        # Viral prediction
        viral_probability = gokai_score.weighted_score * 100
        if viral_probability > 75:
            print(f"   🚀 VIRAL POTENTIAL: {viral_probability:.1f}% - EXTREMELY HIGH! 🔥🔥🔥")
        elif viral_probability > 60:
            print(f"   📈 VIRAL POTENTIAL: {viral_probability:.1f}% - High potential! 🔥")
        elif viral_probability > 40:
            print(f"   📊 VIRAL POTENTIAL: {viral_probability:.1f}% - Moderate potential")
        else:
            print(f"   💡 VIRAL POTENTIAL: {viral_probability:.1f}% - Needs optimization")
        
        time.sleep(1)

def demonstrate_pipeline_power():
    """Demonstrate the full pipeline processing power."""
    print("\n" + "="*80)
    print("⚡ PIPELINE POWER DEMONSTRATION")
    print("="*80)
    
    print("\n🎬 Processing multiple content types simultaneously...")
    
    # Prepare texts with contexts
    texts = list(DEMO_TEXTS.values())
    contexts = [
        {'content_type': name, 'analysis_timestamp': datetime.now().isoformat()}
        for name in DEMO_TEXTS.keys()
    ]
    
    print(f"\n⏱️  Processing {len(texts)} texts in parallel...")
    start_time = time.time()
    
    # Batch processing
    results = batch_execute(
        texts=texts,
        contexts=contexts,
        mode="hybrid",
        parallel=True,
        save_to_memory=True
    )
    
    processing_time = time.time() - start_time
    
    print("\n🚀 BATCH PROCESSING COMPLETE!")
    print(f"   ⚡ Processed {len(texts)} texts in {processing_time:.2f} seconds")
    print(f"   📊 Average: {processing_time/len(texts)*1000:.1f}ms per text")
    print(f"   💪 Processing Speed: {len(texts)/processing_time:.1f} texts/second")
    
    # Show results summary
    successful = sum(1 for r in results if r['success'])
    decisions = {}
    total_quantum = 0
    total_gokai = 0
    
    for result in results:
        if result['success']:
            decision = result['decision']
            decisions[decision] = decisions.get(decision, 0) + 1
            total_quantum += result['quantum_score']
            total_gokai += result['gokai_score']
    
    print("\n📈 PROCESSING SUMMARY:")
    print(f"   ✅ Success Rate:     {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"   🎯 Avg Quantum:      {total_quantum/successful:.3f}")
    print(f"   💎 Avg GOKAI:        {total_gokai/successful:.3f}")
    print("   ⚖️  Decision Breakdown:")
    
    for decision, count in decisions.items():
        percentage = count / successful * 100
        print(f"      {decision.title():>10}: {count:>2} ({percentage:>5.1f}%)")

def demonstrate_real_time_analysis():
    """Demonstrate real-time analysis capabilities."""
    print("\n" + "="*80)
    print("🔴 REAL-TIME ANALYSIS SIMULATION")
    print("="*80)
    
    print("\n🎥 Simulating real-time content analysis...")
    
    # Simulate streaming content
    streaming_content = [
        "Just discovered quantum text analysis! 🤯",
        "This changes everything we know about content creation.",
        "The mathematics behind viral content is fascinating!",
        "AI + Quantum Physics + Text Analysis = Mind Blown 🧠💥",
        "Can't believe how accurate these predictions are!"
    ]
    
    memory = MemoryPersistence(':memory:')  # In-memory database for demo
    
    print("\n📡 Processing incoming content stream...\n")
    
    for i, content in enumerate(streaming_content, 1):
        print(f"📨 Incoming [{i}/{len(streaming_content)}]: {content}")
        
        # Real-time processing
        start = time.time()
        result = execute(
            text=content,
            context={
                'real_time': True,
                'stream_id': i,
                'timestamp': datetime.now().isoformat()
            },
            mode="quantum"
        )
        process_time = (time.time() - start) * 1000
        
        if result['success']:
            print(f"   ⚡ Processed in {process_time:.1f}ms")
            print(f"   🎯 Score: {result['quantum_score']:.3f} | Decision: {result['decision']}")
            
            # Store in memory
            memory.save_memory({
                'content': content,
                'result': result,
                'processing_time_ms': process_time
            }, memory_type='realtime_analysis')
            
            # Real-time feedback
            if result['quantum_score'] > 0.7:
                print("   🔥 HIGH ENGAGEMENT POTENTIAL!")
            elif result['decision'] == 'accept':
                print("   ✅ Good content quality")
            elif result['decision'] == 'review':
                print("   🔍 Needs review")
            else:
                print("   💡 Improvement recommended")
        
        print()
        time.sleep(0.8)  # Simulate real-time delay
    
    # Show memory statistics
    stats = memory.get_statistics()
    print("📊 REAL-TIME SESSION SUMMARY:")
    print(f"   💾 Stored {stats['total_memories']} analysis results")
    print(f"   ⏱️  Average processing time: {sum(r['processing_time_ms'] for r in [memory.get_memory(i+1)['data'] for i in range(len(streaming_content))])/len(streaming_content):.1f}ms")

def demonstrate_viral_prediction():
    """Demonstrate viral content prediction."""
    print("\n" + "="*80)
    print("🚀 VIRAL CONTENT PREDICTION ENGINE")
    print("="*80)
    
    print("\n🎯 Testing our viral prediction algorithm...")
    
    # Test different content styles
    test_contents = [
        ("🧵 THREAD: The secret mathematical pattern behind ALL viral content (you won't believe #7!)", "High-engagement thread"),
        ("Scientists discover that successful tweets follow Fibonacci sequence patterns.", "Scientific finding"),
        ("Hello everyone, hope you're having a nice day.", "Generic greeting"),
        ("🔥🔥🔥 This AI just PREDICTED my tweet would go viral and IT DID! 1M views in 6 hours! 📈💯", "Viral claim"),
        ("The weather is okay today.", "Simple statement")
    ]
    
    print("\n📊 VIRAL POTENTIAL ANALYSIS:\n")
    
    for content, description in test_contents:
        print(f"📝 Content Type: {description}")
        print(f"💬 Text: {content}")
        
        # Enhanced context for viral analysis
        viral_context = {
            'importance': 0.8,
            'urgency': 0.7,
            'quality': 0.75,
            'engagement_markers': len([c for c in content if c in '🔥💯📈⚡🚀']),
            'caps_ratio': sum(1 for c in content if c.isupper()) / len(content),
            'exclamation_count': content.count('!'),
            'hashtag_count': content.count('#'),
            'emoji_count': len([c for c in content if ord(c) > 127])
        }
        
        result = execute(text=content, context=viral_context, mode="hybrid")
        
        if result['success']:
            # Calculate viral score
            base_score = result['quantum_score'] * result['gokai_score']
            engagement_bonus = min(viral_context['engagement_markers'] * 0.1, 0.3)
            viral_score = min((base_score + engagement_bonus) * result['confidence'], 1.0)
            
            print("\n   🎯 Analysis Results:")
            print(f"      Quantum Score:     {result['quantum_score']:.3f}")
            print(f"      GOKAI Score:       {result['gokai_score']:.3f}")
            print(f"      Confidence:        {result['confidence']:.3f}")
            print(f"      Engagement Markers: {viral_context['engagement_markers']}")
            print(f"      🚀 VIRAL SCORE:     {viral_score:.3f}")
            
            # Viral prediction
            viral_percentage = viral_score * 100
            if viral_percentage >= 80:
                print(f"      🔥 PREDICTION: {viral_percentage:.0f}% - EXTREMELY LIKELY TO GO VIRAL! 🚀🚀🚀")
            elif viral_percentage >= 60:
                print(f"      📈 PREDICTION: {viral_percentage:.0f}% - High viral potential! 🔥")
            elif viral_percentage >= 40:
                print(f"      📊 PREDICTION: {viral_percentage:.0f}% - Moderate viral potential")
            elif viral_percentage >= 20:
                print(f"      💡 PREDICTION: {viral_percentage:.0f}% - Low viral potential")
            else:
                print(f"      😴 PREDICTION: {viral_percentage:.0f}% - Unlikely to go viral")
        
        print("\n" + "-"*70 + "\n")
        time.sleep(1)

def interactive_demo():
    """Interactive demo where users can input their own text."""
    print("\n" + "="*80)
    print("🎮 INTERACTIVE DEMO - TEST YOUR OWN CONTENT!")
    print("="*80)
    
    print("\n✨ Now it's your turn! Enter your own text to see how it scores...")
    print("💡 Tip: Try content with emojis, caps, and engaging language for higher scores!")
    print("🛑 Type 'quit' to exit the interactive demo\n")
    
    while True:
        try:
            user_text = input("📝 Enter your text: ").strip()
            
            if user_text.lower() == 'quit':
                print("\n👋 Thanks for trying the interactive demo!")
                break
                
            if not user_text:
                print("❌ Please enter some text to analyze!\n")
                continue
            
            print(f"\n🔍 Analyzing: '{user_text[:50]}{'...' if len(user_text) > 50 else ''}'")
            
            # Analyze user text
            context = {
                'user_input': True,
                'interactive_demo': True,
                'timestamp': datetime.now().isoformat()
            }
            
            result = execute(text=user_text, context=context, mode="hybrid")
            
            if result['success']:
                print("\n🎯 YOUR RESULTS:")
                print(f"   🌀 Quantum Score:   {result['quantum_score']:.3f}/1.000")
                print(f"   💎 GOKAI Score:     {result['gokai_score']:.3f}/1.000")
                print(f"   🎪 Confidence:      {result['confidence']:.3f}/1.000")
                print(f"   ⚖️  Decision:        {result['decision'].upper()}")
                
                # Fun analysis
                overall_score = (result['quantum_score'] + result['gokai_score']) / 2
                if overall_score >= 0.8:
                    print("   🏆 AMAZING! Your text has exceptional mathematical harmony!")
                elif overall_score >= 0.6:
                    print("   🔥 GREAT! Your text shows strong patterns and engagement potential!")
                elif overall_score >= 0.4:
                    print("   ✅ GOOD! Your text has decent structure and complexity.")
                else:
                    print("   💡 Room for improvement! Try adding more complexity or engaging elements.")
                    
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
            print("\n" + "-"*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Thanks for trying SpiralMind-Nexus!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Please try again with different text.\n")

def show_final_stats():
    """Show final demo statistics and call-to-action."""
    print("\n" + "="*80)
    print("📊 DEMO COMPLETE - FINAL STATISTICS")
    print("="*80)
    
    try:
        from spiral.pipeline.double_pipeline import get_pipeline_statistics
        stats_result = get_pipeline_statistics()
        
        if stats_result['success']:
            stats = stats_result['statistics']
            print("\n🎯 DEMO SESSION STATISTICS:")
            print(f"   📈 Total Texts Processed:    {stats.get('total_processed', 0)}")
            print(f"   ✅ Successful Analyses:      {stats.get('successful_processed', 0)}")
            print(f"   ⚡ Average Processing Time:   {stats.get('average_processing_time', 0):.3f}s")
            print(f"   🎪 Success Rate:             {stats.get('success_rate', 0)*100:.1f}%")
            
            if 'decisions' in stats:
                print("\n⚖️  DECISION BREAKDOWN:")
                for decision, count in stats['decisions'].items():
                    print(f"      {decision.title():>8}: {count:>3}")
        
    except Exception as e:
        print(f"📊 Statistics temporarily unavailable: {e}")
    
    print("\n" + "="*80)
    print("🚀 CONGRATULATIONS! You've experienced the power of SpiralMind-Nexus!")
    print("="*80)
    
    print("\n🎯 What you've seen today:")
    print("   ✅ Quantum text analysis using Fibonacci sequences")
    print("   ✅ GOKAI scoring with confidence metrics")
    print("   ✅ Real-time processing capabilities")
    print("   ✅ Viral content prediction")
    print("   ✅ Batch processing power")
    print("   ✅ Interactive analysis")
    
    print("\n🔗 Next Steps:")
    print("   🛠️  Try the CLI: spiral --text 'Your text here'")
    print("   🌐 Start the API: python -m spiral.api")
    print("   🐳 Use Docker: docker-compose up spiral-api")
    print("   📖 Read the docs: /docs/")
    print("   🧪 Run tests: pytest tests/")
    
    print("\n💬 Ready to revolutionize your text analysis? Let's build the future together!")
    print("🌟 Star us on GitHub and share your experience!")
    
if __name__ == "__main__":
    try:
        print_banner()
        
        # Run all demonstrations
        demonstrate_quantum_processing()
        demonstrate_gokai_scoring()
        demonstrate_pipeline_power()
        demonstrate_real_time_analysis()
        demonstrate_viral_prediction()
        
        # Interactive portion
        interactive_demo()
        
        # Final statistics
        show_final_stats()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted. Thanks for checking out SpiralMind-Nexus!")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Demo encountered an error: {e}")
        print("💡 Please check your installation and try again.")
    
    print("\n🎉 Thanks for experiencing SpiralMind-Nexus!")
    print("💫 Keep exploring the quantum realm of text analysis!")
