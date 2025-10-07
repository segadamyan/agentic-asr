#!/usr/bin/env python3
"""
Comprehensive reranker comparison script.
Retrieves data from database, tests all rerankers, and shows comparison table.
"""

import logging
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from tabulate import tabulate
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_asr.config import Config
from agentic_asr.rerankers import get_reranker

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during comparison
logger = logging.getLogger(__name__)


class RerankerComparator:
    """Compare different rerankers with real data from database."""
    
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self.vector_store = None
        self.test_queries = []
        
    def load_data_from_vector_store(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Load data from existing vector store for a specific query."""
        print(f"Loading data from vector store for query: '{query[:50]}...'")
        
        from agentic_asr.vector_store import get_vector_store
        
        # Initialize vector store
        vector_store = get_vector_store()
        vector_store.initialize()
        
        # Get statistics
        stats = vector_store.get_stats()
        print(f"Vector store contains {stats['total_documents']} documents with {stats['total_chunks']} chunks")
        
        if stats['total_chunks'] == 0:
            raise ValueError("No data in vector store. Please add some data first.")
        
        # Search for chunks relevant to the specific query
        results = []
        seen_chunks = set()  # Track chunks we've already added
        
        try:
            # Primary search with the actual query
            search_results = vector_store.search(
                query=query,
                top_k=limit * 2,  # Get more results to account for deduplication
                similarity_threshold=0.1,
                use_reranking=False  # We'll test reranking separately
            )
            
            for result in search_results:
                chunk_id = result.get("chunk_id", "unknown")
                
                # Skip if we've already seen this chunk
                if chunk_id in seen_chunks:
                    continue
                
                # Add to seen set
                seen_chunks.add(chunk_id)
                
                # Create a display name that includes chunk info
                filename = result.get("filename", "unknown")
                display_name = f"{filename}[{chunk_id}]"
                
                results.append({
                    "chunk_id": chunk_id,
                    "content": result["content"],
                    "filename": filename,
                    "display_name": display_name,
                    "language": "armenian",  # Based on your data
                    "similarity_score": result["similarity_score"]
                })
                
                if len(results) >= limit:
                    break
                    
        except Exception as e:
            print(f"Error searching for query '{query[:30]}...': {e}")
        
        if not results:
            raise ValueError(f"No relevant content found for query: '{query[:50]}...'. Please check if the vector store contains relevant data.")
        
        # If we don't have enough diverse results, try a broader search
        if len(results) < limit:
            print(f"Only found {len(results)} unique chunks for query, trying broader search...")
            try:
                broader_results = vector_store.search(
                    query="",  # Empty query to get most diverse results
                    top_k=limit * 2,
                    similarity_threshold=0.0,  # Lower threshold
                    use_reranking=False
                )
                
                for result in broader_results:
                    chunk_id = result.get("chunk_id", "unknown")
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        filename = result.get("filename", "unknown")
                        display_name = f"{filename}[{chunk_id}]"
                        
                        results.append({
                            "chunk_id": chunk_id,
                            "content": result["content"],
                            "filename": filename,
                            "display_name": display_name,
                            "language": "armenian",
                            "similarity_score": result["similarity_score"]
                        })
                        
                        if len(results) >= limit:
                            break
            except Exception as e:
                print(f"Broader search failed: {e}")
        
        print(f"Loaded {len(results)} unique chunks for query")
        print(f"Top chunk IDs: {[r.get('chunk_id', '?') for r in results[:10]]}")
        return results[:limit]
    


    def get_test_queries(self) -> List[Dict[str, str]]:
        """Get AI-related test queries in different languages."""
        return [
            {
                "query": "Ինչպե՞ս է արհեստական բանականության էժանացումը ազդելու ծրագրավորողների աշխատաշուկայի և աշխատավարձերի վրա։",
                "language": "armenian",
                "description": "Armenian: AI impact on programmer job market and salaries"
            },
            {
                "query": "Ինչպե՞ս է արհեստական բանականությունը փոխում բիզնեսների թվայնացման արժեքը և շուկայի պահանջարկը։",
                "language": "armenian",
                "description": "Armenian: AI changing business digitization value and market demand"
            },
            {
                "query": "Ի՞նչ սոցիալական վտանգներ և վարքային փոփոխություններ կարող են առաջանալ, երբ մարդիկ սկսեն շփվել արհեստական ընկերների հետ։",
                "language": "armenian",
                "description": "Armenian: Social dangers and behavioral changes from AI companions"
            },
            {
                "query": "Ինչո՞ւ է արհեստական բանականության հեղափոխությունը հաճախ համեմատվում ԽՍՀՄ փլուզման հետ՝ ադապտացման տեսանկյունից։",
                "language": "armenian",
                "description": "Armenian: AI revolution compared to USSR collapse in adaptation terms"
            },
            {
                "query": "Ինչպե՞ս է OpenAI-ի մոտեցումը՝ «թողնել մոդելին ավելի երկար մտածել», բարելավում արդյունքների որակը։",
                "language": "armenian", 
                "description": "Armenian: OpenAI approach of letting models think longer"
            },
            {
                "query": "Ինչու՞ է GPU-ների զարգացումը դարձել կենտրոնական գործոն արհեստական բանականության առաջխաղացման մեջ։",
                "language": "armenian",
                "description": "Armenian: GPU development as central factor in AI advancement"
            },
            {
                "query": "Ինչպե՞ս է արհեստական բանականությունը փոխելու կրթական համակարգը և մասնագիտական աճի մոդելները։",
                "language": "armenian",
                "description": "Armenian: AI changing education systems and professional growth models"
            },
            {
                "query": "Ինչո՞ւ է գրել սովորելը համեմատվում սպորտային մարզումների հետ՝ որպես մտավոր հմտություն պահպանելու միջոց։",
                "language": "armenian",
                "description": "Armenian: Learning to write compared to sports training as intellectual skill preservation"
            },
            {
                "query": "Ինչպե՞ս կարող է արհեստական բանականությունը բարձրացնել պետական կառավարման արդյունավետությունը և կրճատել բյուրոկրատիան։",
                "language": "armenian",
                "description": "Armenian: AI improving government efficiency and reducing bureaucracy"
            },
            {
                "query": "Ի՞նչ տնտեսական և կառավարման հետևանքներ կարող են առաջանալ, երբ ծրագրային ապահովումը դառնում է էժան և հասանելի բոլորին։",
                "language": "armenian",
                "description": "Armenian: Economic and governance consequences when software becomes cheap and accessible"
            }
        ]
    
    def get_available_rerankers(self) -> List[Dict[str, Any]]:
        """Get list of available rerankers with their configs."""
        rerankers = [
            {"type": "none", "kwargs": {}, "name": "No Reranker"},
            {"type": "bm25", "kwargs": {}, "name": "BM25"},
            {"type": "keyword_boost", "kwargs": {"boost_factor": 1.5}, "name": "Keyword Boost"},
        ]
        
        rerankers.extend([
            {
                "type": "cross_encoder", 
                "kwargs": {"model_name": "cross-encoder/ms-marco-TinyBERT-L-2-v2"},
                "name": "Cross Encoder"
            },
            {
                "type": "multilingual_cross_encoder",
                "kwargs": {
                    "model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                    "armenian_boost": 1.2
                },
                "name": "Multilingual Cross Encoder"
            }
        ])
        
        # Check if LLM is available
        try:
            from agentic_asr.config import Config
            if Config.ANTHROPIC_API_KEY or Config.OPENAI_API_KEY:
                rerankers.append({
                    "type": "llm_agent",
                    "kwargs": {
                        "provider_name": "anthropic" if Config.ANTHROPIC_API_KEY else "openai",
                        "model": "claude-sonnet-4-20250514" if Config.ANTHROPIC_API_KEY else "gpt-5",
                        "max_chunks_per_batch": 3  # Limit for cost control
                    },
                    "name": "LLM Agent"
                })
            else:
                print("Note: No LLM API keys found. LLM agent reranker will be skipped.")
        except Exception as e:
            print(f"Note: LLM components not available. LLM agent reranker will be skipped. ({e})")
        
        return rerankers
    
    def _get_expected_class_name(self, reranker_type: str) -> str:
        """Get the expected class name for a reranker type."""
        type_to_class = {
            "none": "NoReranker",
            "bm25": "BM25Reranker", 
            "keyword_boost": "KeywordBoostReranker",
            "cross_encoder": "CrossEncoderReranker",
            "multilingual_cross_encoder": "MultilingualCrossEncoderReranker",
            "llm_agent": "LLMAgentReranker"
        }
        return type_to_class.get(reranker_type, f"{reranker_type}Reranker")

    def evaluate_reranker(
        self,
        reranker_config: Dict[str, Any],
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a single reranker."""
        reranker_name = reranker_config["name"]
        
        try:
            start_time = time.time()
            
            # Initialize reranker
            reranker = get_reranker(reranker_config["type"], **reranker_config["kwargs"])
            
            # Only show warning for actual fallbacks (like LLM -> BM25)
            if hasattr(reranker, 'name') and "fallback" in reranker.name:
                reranker_name = f"{reranker_config['name']} (→ BM25)"
            else:
                reranker_name = reranker_config["name"]
            
            # Rerank results
            reranked = reranker.rerank(query, results.copy(), top_k=5)
            
            end_time = time.time()
            
            # Calculate metrics
            armenian_results = sum(1 for r in reranked[:3] if self._is_armenian_text(r["content"]))
            avg_score = sum(r.get("similarity_score", 0) for r in reranked[:3]) / min(3, len(reranked))
            
            return {
                "name": reranker_name,
                "top_3_results": [r.get("display_name", f"{r['filename']}[{r.get('chunk_id', '?')}]") for r in reranked[:3]],
                "armenian_in_top_3": armenian_results,
                "avg_similarity": avg_score,
                "processing_time_ms": round((end_time - start_time) * 1000, 2),
                "total_results": len(reranked),
                "reranked_results": reranked[:3]  # Store for detailed analysis
            }
            
        except Exception as e:
            print(f"ERROR in {reranker_name}: {str(e)}")
            return {
                "name": f"{reranker_name} (ERROR)",
                "error": str(e),
                "top_3_results": [],
                "armenian_in_top_3": 0,
                "avg_similarity": 0,
                "processing_time_ms": 0,
                "total_results": 0
            }
    
    def _is_armenian_text(self, text: str) -> bool:
        """Check if text contains Armenian characters."""
        armenian_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if (0x0530 <= ord(char) <= 0x058F) or (0xFB13 <= ord(char) <= 0xFB17):
                    armenian_chars += 1
        
        return (armenian_chars / total_chars) > 0.1 if total_chars > 0 else False
    
    def run_comparison(self):
        """Run the full comparison."""
        print("🔍 Reranker Comparison Tool")
        print("=" * 50)
        
        # Get test queries
        test_queries = self.get_test_queries()
        
        # Get available rerankers
        rerankers = self.get_available_rerankers()
        
        print(f"\nTesting {len(rerankers)} rerankers with {len(test_queries)} queries")
        print(f"Loading fresh data from vector store for each query\n")
        
        # Run comparisons
        all_results = []
        
        for query_info in test_queries:
            query = query_info["query"]
            query_lang = query_info["language"]
            
            print(f"\n📋 Query: '{query}' ({query_info['description']})")
            print("-" * 60)
            
            # Load data specifically for this query
            try:
                query_data = self.load_data_from_vector_store(query, limit=10)
                if not query_data:
                    print(f"No data found for query, skipping...")
                    continue
            except Exception as e:
                print(f"Error loading data for query: {e}")
                continue
            
            query_results = []
            
            for reranker_config in rerankers:
                result = self.evaluate_reranker(reranker_config, query, query_data)
                result["query"] = query
                result["query_language"] = query_lang
                query_results.append(result)
                all_results.append(result)
            
            # Show table for this query
            self._print_query_table(query_results)
            
            # Show detailed content for the first reranker to see what we're working with
            if query_results and "reranked_results" in query_results[0]:
                print(f"\n📄 Content Preview (using {query_results[0]['name']} reranker):")
                self._print_content_preview(query_results[0]["reranked_results"])
        
        # Show overall summary
        print(f"\n📊 OVERALL SUMMARY")
        print("=" * 70)
        self._print_summary_table(all_results)
        
        # Show consolidated comparison table
        print(f"\n📋 CONSOLIDATED COMPARISON TABLE")
        print("=" * 80)
        self._print_consolidated_table(all_results, test_queries, rerankers)
        
        # Show simplified chunk ranking table
        print(f"\n🎯 CHUNK RANKING COMPARISON")
        print("=" * 60)
        self._print_chunk_ranking_table(all_results, test_queries, rerankers)
        
        # Show detailed analysis for Armenian queries
        armenian_results = [r for r in all_results if r["query_language"] == "armenian"]
        if armenian_results:
            print(f"\n🇦🇲 ARMENIAN QUERY ANALYSIS")
            print("=" * 50)
            self._print_armenian_analysis(armenian_results)
    
    def _print_query_table(self, results: List[Dict[str, Any]]):
        """Print comparison table for a single query."""
        table_data = []
        for result in results:
            if "error" in result:
                table_data.append([
                    result["name"],
                    f"ERROR: {result['error'][:30]}...",
                    "0",
                    "0.000",
                    "0"
                ])
            else:
                # Shorten display names for better table formatting
                top_3_display = []
                for name in result["top_3_results"]:
                    if len(name) > 40:
                        # Show just filename and chunk ID
                        if '[' in name and ']' in name:
                            base_name = name.split('[')[0]
                            chunk_part = '[' + name.split('[')[1]
                            if len(base_name) > 25:
                                base_name = "..." + base_name[-22:]
                            name = base_name + chunk_part
                        else:
                            name = "..." + name[-37:]
                    top_3_display.append(name)
                
                table_data.append([
                    result["name"],
                    "\n".join(top_3_display),  # Each result on a new line for better readability
                    result["armenian_in_top_3"],
                    f"{result['avg_similarity']:.3f}",
                    f"{result['processing_time_ms']}"
                ])
        
        headers = ["Reranker", "Top 3 Chunks", "Armenian", "Avg Score", "Time (ms)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def _print_content_preview(self, results: List[Dict[str, Any]]):
        """Print content preview of top results."""
        for i, result in enumerate(results[:3], 1):
            display_name = result.get("display_name", f"{result['filename']}[{result.get('chunk_id', '?')}]")
            content = result["content"]
            
            # Truncate content for preview
            if len(content) > 100:
                content = content[:100] + "..."
            
            print(f"  {i}. {display_name}")
            print(f"     {content}")
            print(f"     Score: {result.get('similarity_score', 0):.3f}")
            print()
    
    def _print_summary_table(self, all_results: List[Dict[str, Any]]):
        """Print overall summary table."""
        # Group by reranker
        reranker_stats = {}
        
        for result in all_results:
            name = result["name"]
            if name not in reranker_stats:
                reranker_stats[name] = {
                    "total_queries": 0,
                    "total_time": 0,
                    "total_armenian": 0,
                    "total_score": 0,
                    "errors": 0
                }
            
            stats = reranker_stats[name]
            stats["total_queries"] += 1
            
            if "error" not in result:
                stats["total_time"] += result["processing_time_ms"]
                stats["total_armenian"] += result["armenian_in_top_3"]
                stats["total_score"] += result["avg_similarity"]
            else:
                stats["errors"] += 1
        
        table_data = []
        for name, stats in reranker_stats.items():
            if stats["total_queries"] > stats["errors"]:
                avg_time = stats["total_time"] / (stats["total_queries"] - stats["errors"])
                avg_armenian = stats["total_armenian"] / (stats["total_queries"] - stats["errors"])
                avg_score = stats["total_score"] / (stats["total_queries"] - stats["errors"])
                
                table_data.append([
                    name,
                    f"{avg_time:.1f}",
                    f"{avg_armenian:.1f}",
                    f"{avg_score:.3f}",
                    stats["errors"]
                ])
            else:
                table_data.append([name, "N/A", "N/A", "N/A", stats["errors"]])
        
        headers = ["Reranker", "Avg Time (ms)", "Avg Armenian", "Avg Score", "Errors"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def _print_consolidated_table(self, all_results: List[Dict[str, Any]], test_queries: List[Dict[str, str]], rerankers: List[Dict[str, Any]]):
        """Print consolidated table with queries as rows and rerankers as columns."""
        
        # Create a mapping of query -> reranker -> result
        query_reranker_map = {}
        for result in all_results:
            query = result["query"]
            reranker = result["name"]
            
            if query not in query_reranker_map:
                query_reranker_map[query] = {}
            
            query_reranker_map[query][reranker] = result
        
        # Prepare table data
        table_data = []
        reranker_names = [r["name"] for r in rerankers]
        
        for query_info in test_queries:
            query = query_info["query"]
            query_short = query_info["description"].split(": ")[1] if ": " in query_info["description"] else query[:30]
            
            row = [f"{query_short} ({query_info['language'][:2].upper()})"]
            
            for reranker_name in reranker_names:
                if query in query_reranker_map and reranker_name in query_reranker_map[query]:
                    result = query_reranker_map[query][reranker_name]
                    
                    if "error" in result:
                        row.append("ERROR")
                    else:
                        # Show top chunk ID and score
                        if result["top_3_results"]:
                            top_chunk = result["top_3_results"][0]
                            # Extract chunk ID from display name
                            chunk_id = "?"
                            if '[' in top_chunk and ']' in top_chunk:
                                chunk_id = top_chunk.split('[')[1].split(']')[0]
                            
                            score = result["avg_similarity"]
                            time_ms = result["processing_time_ms"]
                            
                            # Format: chunk[ID] (score/time)
                            row.append(f"[{chunk_id}]\n({score:.2f}/{time_ms:.0f}ms)")
                        else:
                            row.append("No results")
                else:
                    row.append("N/A")
            
            table_data.append(row)
        
        headers = ["Query"] + reranker_names
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print("\nFormat: [chunk_id] (avg_score/time_ms)")
        print("- chunk_id: ID of top-ranked chunk")
        print("- avg_score: Average similarity score of top 3 results")
        print("- time_ms: Processing time in milliseconds")
    
    def _print_chunk_ranking_table(self, all_results: List[Dict[str, Any]], test_queries: List[Dict[str, str]], rerankers: List[Dict[str, Any]]):
        """Print simplified table showing just top chunk IDs for each query-reranker combination."""
        
        # Create a mapping of query -> reranker -> result
        query_reranker_map = {}
        for result in all_results:
            query = result["query"]
            reranker = result["name"]
            
            if query not in query_reranker_map:
                query_reranker_map[query] = {}
            
            query_reranker_map[query][reranker] = result
        
        # Prepare table data
        table_data = []
        reranker_names = [r["name"] for r in rerankers]
        
        for query_info in test_queries:
            query = query_info["query"]
            # Create shorter query description
            lang_code = query_info["language"][:2].upper()
            if ": " in query_info["description"]:
                query_short = query_info["description"].split(": ")[1][:25]
            else:
                query_short = query[:25]
            
            if len(query_short) > 25:
                query_short = query_short[:22] + "..."
            
            row = [f"{query_short} ({lang_code})"]
            
            for reranker_name in reranker_names:
                if query in query_reranker_map and reranker_name in query_reranker_map[query]:
                    result = query_reranker_map[query][reranker_name]
                    
                    if "error" in result:
                        row.append("ERR")
                    else:
                        # Show just the top chunk ID
                        if result["top_3_results"]:
                            top_chunk = result["top_3_results"][0]
                            # Extract chunk ID from display name
                            chunk_id = "?"
                            if '[' in top_chunk and ']' in top_chunk:
                                chunk_id = top_chunk.split('[')[1].split(']')[0]
                            row.append(f"[{chunk_id}]")
                        else:
                            row.append("[-]")
                else:
                    row.append("N/A")
            
            table_data.append(row)
        
        headers = ["Query"] + [name.replace(" ", "\n") for name in reranker_names]  # Multi-line headers
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print("\nLegend:")
        print("- [XX]: Chunk ID of top-ranked result")
        print("- ERR: Error occurred during reranking")
        print("- [-]: No results returned")
        print("- Different chunk IDs indicate different reranking behavior")
    
    def _print_armenian_analysis(self, armenian_results: List[Dict[str, Any]]):
        """Print detailed analysis for Armenian queries."""
        print("Performance with Armenian queries:")
        
        # Group by reranker
        reranker_armenian = {}
        for result in armenian_results:
            name = result["name"]
            if name not in reranker_armenian:
                reranker_armenian[name] = []
            reranker_armenian[name].append(result["armenian_in_top_3"])
        
        table_data = []
        for name, armenian_counts in reranker_armenian.items():
            avg_armenian = sum(armenian_counts) / len(armenian_counts)
            max_armenian = max(armenian_counts)
            table_data.append([
                name,
                f"{avg_armenian:.1f}",
                f"{max_armenian}",
                len([x for x in armenian_counts if x > 0])
            ])
        
        headers = ["Reranker", "Avg Armenian/Top3", "Max Armenian", "Queries w/Armenian"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        best_for_armenian = max(reranker_armenian.items(), 
                               key=lambda x: sum(x[1]) / len(x[1]))
        print(f"• Best for Armenian content: {best_for_armenian[0]}")
        print("• Cross-encoder models provide more accurate semantic matching")
        print("• Multilingual cross-encoder specifically boosts Armenian content")


def main():
    """Main function."""
    try:
        comparator = RerankerComparator()
        comparator.run_comparison()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
