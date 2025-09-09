#edge_builder.py
from typing import Dict, List, Tuple, Set, Optional
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import QAPair, QADataset

class EdgeBuilder:
    """
    A class for building and managing different types of edges between QAPair objects.
    Maintains separate adjacency lists for different edge methods and provides 
    visualization capabilities.
    """
    
    def __init__(self, dataset: QADataset):
        """
        Initialize the EdgeBuilder with a QADataset.
        
        Args:
            dataset: The QADataset containing QAPairs to build edges for
        """
        self.dataset = dataset
        
        # Dictionary to store adjacency lists for different methods
        # Format: {method_name: {source_id: [target_id1, target_id2, ...]}}
        self.adjacency_lists: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        # Set to keep track of all methods used
        self.methods: Set[str] = set()
        
        # Load existing edges from the dataset
        self._initialize_adjacency_lists()
    
    def _initialize_adjacency_lists(self):
        """
        Initialize adjacency lists based on existing edges in the dataset.
        """
        for qa_pair in self.dataset:
            for method_name, target_id in qa_pair.edges:
                self.add_edge(qa_pair.id, target_id, method_name)
    
    def add_edge(self, source_id: int, target_id: int, method_name: str = "default"):
        """
        Add an edge between source and target with the specified method.
        Updates both the internal adjacency list and the dataset.
        
        Args:
            source_id: ID of the source QAPair
            target_id: ID of the target QAPair
            method_name: Name of the method used to create this edge
        """
        # Add to adjacency list
        if target_id not in self.adjacency_lists[method_name][source_id]:
            self.adjacency_lists[method_name][source_id].append(target_id)
        
        # Update methods set
        self.methods.add(method_name)
        
        # Update the edge in the dataset
        self.dataset.add_edge(source_id, target_id, method_name)
    
    def build_cosine_similarity_edges(self, threshold: float = 0.2, method_name: str = "cosine_similarity"):
        """
        Build edges between QA pairs based on cosine similarity of their combined text.
        
        Args:
            threshold: Minimum similarity score required to create an edge
            method_name: Name to assign to this edge method
        
        Returns:
            Number of new edges created
        """
        # Prepare the text data
        qa_ids = []
        qa_texts = []
        
        for qa_pair in self.dataset:
            qa_ids.append(qa_pair.id)
            
            # Combine content, removing None values
            parts = []
            if qa_pair.context:
                parts.append(qa_pair.context)
            if qa_pair.question:
                parts.append(qa_pair.question)
            if qa_pair.answer:
                parts.append(qa_pair.answer)
            
            combined_text = " ".join(parts)
            qa_texts.append(combined_text if combined_text else "")
        
        # Calculate TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(qa_texts)
            cosine_sim = cosine_similarity(tfidf_matrix)
        except ValueError:
            # Handle the case where qa_texts may contain only empty strings
            print("Warning: Could not calculate similarities due to empty texts.")
            return 0
        
        # Create edges based on similarity threshold
        edges_created = 0
        for i in range(len(qa_ids)):
            for j in range(i+1, len(qa_ids)):  # Only upper triangle to avoid duplicates
                if cosine_sim[i, j] >= threshold:
                    # Create bidirectional edges
                    self.add_edge(qa_ids[i], qa_ids[j], method_name)
                    self.add_edge(qa_ids[j], qa_ids[i], method_name)
                    edges_created += 2
        return edges_created
    
    def build_keyword_overlap_edges(self, threshold: int = 2, method_name: str = "keyword_overlap"):
        """
        Build edges based on the number of overlapping keywords between QA pairs.
        
        Args:
            threshold: Minimum number of overlapping keywords required to create an edge
            method_name: Name to assign to this edge method
            
        Returns:
            Number of new edges created
        """
        edges_created = 0
        
        # Get all QA pairs
        qa_pairs = list(self.dataset)
        
        for i, qa1 in enumerate(qa_pairs):
            for j in range(i+1, len(qa_pairs)):
                qa2 = qa_pairs[j]
                
                # Combine all keywords
                keywords1 = set(qa1.context_keywords + qa1.question_keywords + qa1.answer_keywords)
                keywords2 = set(qa2.context_keywords + qa2.question_keywords + qa2.answer_keywords)
                
                # Calculate overlap
                overlap = len(keywords1.intersection(keywords2))
                
                if overlap >= threshold:
                    # Create bidirectional edges
                    self.add_edge(qa1.id, qa2.id, method_name)
                    self.add_edge(qa2.id, qa1.id, method_name)
                    edges_created += 2
        
        return edges_created
    
    def _bfs_component(self, start_node: int, adjacency_dict: Dict[int, List[int]], visited: Set[int]) -> List[int]:
        """
        Perform BFS to find all nodes in the connected component starting from start_node.
        
        Args:
            start_node: The starting node for BFS
            adjacency_dict: Adjacency dictionary for the graph
            visited: Set of already visited nodes
            
        Returns:
            List of nodes in the connected component
        """
        component = []
        queue = deque([start_node])
        visited.add(start_node)
        
        while queue:
            current_node = queue.popleft()
            component.append(current_node)
            
            # Check outgoing edges
            for neighbor in adjacency_dict.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
            
            # Check incoming edges (since we want undirected connectivity)
            for node_id, neighbors in adjacency_dict.items():
                if current_node in neighbors and node_id not in visited:
                    visited.add(node_id)
                    queue.append(node_id)
        
        return component
    
    def get_rare_data_nodes(self, k: int = 3, method_name: Optional[str] = None) -> Tuple[List[int], Dict[str, List[List[int]]]]:
        """
        Find nodes that belong to small connected components (size <= k) in the graph.
        These nodes represent rare or isolated data points.
        
        Args:
            k: Maximum size of connected components to consider as "rare"
            method_name: If provided, only use edges from this method. 
                        If None, combine all methods.
        
        Returns:
            Tuple containing:
            - List of rare node IDs
            - Dictionary mapping method names to lists of small components
        """
        rare_nodes = []
        component_details = {}
        
        # Determine which methods to use
        methods_to_use = [method_name] if method_name else list(self.methods)
        
        if not methods_to_use:
            print("No edge methods available")
            return [], {}
        
        for method in methods_to_use:
            if method not in self.adjacency_lists:
                print(f"Method '{method}' not found in adjacency lists")
                continue
            
            # Get adjacency dictionary for this method
            adjacency_dict = self.adjacency_lists[method]
            
            # Get all node IDs from the dataset
            all_node_ids = {qa_pair.id for qa_pair in self.dataset}
            
            visited = set()
            small_components = []
            
            # Find all connected components using BFS
            for node_id in all_node_ids:
                if node_id not in visited:
                    component = self._bfs_component(node_id, adjacency_dict, visited)
                    
                    # Check if component size is <= k
                    if len(component) <= k:
                        small_components.append(component)
                        rare_nodes.extend(component)
            
            component_details[method] = small_components
        
        # Remove duplicates while preserving order
        rare_nodes = list(dict.fromkeys(rare_nodes))
        
        return rare_nodes, component_details
    
    def get_rare_data_qa_pairs(self, k: int = 3, method_name: Optional[str] = None) -> List[QAPair]:
        """
        Get the actual QAPair objects for rare data nodes.
        
        Args:
            k: Maximum size of connected components to consider as "rare"
            method_name: If provided, only use edges from this method
        
        Returns:
            List of QAPair objects that are considered rare data
        """
        rare_node_ids, _ = self.get_rare_data_nodes(k, method_name)
        
        # Create a mapping from ID to QAPair for efficient lookup
        id_to_qa = {qa_pair.id: qa_pair for qa_pair in self.dataset}
        
        # Get the corresponding QAPair objects
        rare_qa_pairs = []
        for node_id in rare_node_ids:
            if node_id in id_to_qa:
                rare_qa_pairs.append(id_to_qa[node_id])
        
        return rare_qa_pairs
    
    def get_edges_by_method(self, method_name: str) -> Dict[int, List[int]]:
        """
        Get all edges for a specific method.
        
        Args:
            method_name: Name of the method to get edges for
            
        Returns:
            Adjacency list for the specified method
        """
        return dict(self.adjacency_lists[method_name])
    
    def get_all_methods(self) -> List[str]:
        """
        Get a list of all edge methods currently in use.
        
        Returns:
            List of method names
        """
        return list(self.methods)
    
    def visualize_graph(self, method_name: Optional[str] = None, output_dir: str = "./graphs"):
        """
        Visualize the graph(s) using NetworkX and Matplotlib.
        
        Args:
            method_name: If provided, visualize only this method's graph.
                         If None, visualize all methods.
            output_dir: Directory to save the graph visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        methods_to_visualize = [method_name] if method_name else self.methods
        
        for method in methods_to_visualize:
            if method not in self.adjacency_lists:
                print(f"No edges found for method: {method}")
                continue
                
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes with attributes
            for qa_pair in self.dataset:
                # Determine node color based on QA type
                node_type = qa_pair.classify()
                
                # Define color map for different QA types
                color_map = {
                    "Document": "skyblue",
                    "Question": "yellow",
                    "Ground Truth": "green",
                    "Context + Question": "orange",
                    "Context + Ground Truth": "purple",
                    "QA Pair": "pink",
                    "Complete QA": "red",
                    "Unknown": "gray"
                }
                
                color = color_map.get(node_type, "gray")
                G.add_node(qa_pair.id, color=color, label=f"ID: {qa_pair.id}\nType: {node_type}")
            
            # Add edges
            for source_id, targets in self.adjacency_lists[method].items():
                for target_id in targets:
                    G.add_edge(source_id, target_id)
            
            # Extract node colors for visualization
            node_colors = [G.nodes[node].get('color', 'gray') for node in G.nodes()]
            
            # Create the figure
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42)  # Consistent layout
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, arrows=True, width=1.0, alpha=0.5)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            # Add title and legend
            plt.title(f"QA Graph - Method: {method}")
            
            # Create legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10, label=qa_type)
                               for qa_type, color in color_map.items()]
            plt.legend(handles=legend_elements, loc='upper right')
            
            # Save the figure
            filename = os.path.join(output_dir, f"graph_{method}.png")
            plt.savefig(filename)
            plt.close()
            
            print(f"Graph for method '{method}' saved to {filename}")
    
    def visualize_all_graphs(self, output_dir: str = "./graphs"):
        """
        Visualize all graphs, one for each method.
        
        Args:
            output_dir: Directory to save the graph visualizations
        """
        for method in self.methods:
            self.visualize_graph(method, output_dir)
    
    def visualize_combined_graph(self, output_file: str = "./graphs/combined_graph.png"):
        """
        Visualize a combined graph with edges from all methods, using different colors.
        
        Args:
            output_file: File path to save the combined graph
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create NetworkX graph
        G = nx.MultiDiGraph()  # Use MultiDiGraph to allow multiple edges between nodes
        
        # Add nodes with attributes
        for qa_pair in self.dataset:
            node_type = qa_pair.classify()
            
            # Define color map for different QA types
            color_map = {
                "Document": "skyblue",
                "Question": "yellow",
                "Ground Truth": "green",
                "Context + Question": "orange",
                "Context + Ground Truth": "purple",
                "QA Pair": "pink",
                "Complete QA": "red",
                "Unknown": "gray"
            }
            
            color = color_map.get(node_type, "gray")
            G.add_node(qa_pair.id, color=color, label=f"ID: {qa_pair.id}\nType: {node_type}")
        
        # Generate a colormap for methods
        method_colors = plt.cm.rainbow(np.linspace(0, 1, len(self.methods)))
        method_color_map = {method: method_colors[i] for i, method in enumerate(self.methods)}
        
        # Add edges with method-specific colors
        for method_idx, method in enumerate(self.methods):
            for source_id, targets in self.adjacency_lists[method].items():
                for target_id in targets:
                    G.add_edge(source_id, target_id, color=method_color_map[method], method=method)
        
        # Extract node colors for visualization
        node_colors = [G.nodes[node].get('color', 'gray') for node in G.nodes()]
        
        # Create the figure
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
        
        # Draw edges by method
        for method in self.methods:
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('method') == method]
            if edges:
                nx.draw_networkx_edges(
                    G, pos, 
                    edgelist=edges, 
                    width=1.0, 
                    alpha=0.6,
                    edge_color=[method_color_map[method]] * len(edges),
                    label=method
                )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Add title and legend
        plt.title("Combined QA Graph - All Methods")
        
        # Create node type legend
        node_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=color, markersize=10, label=qa_type)
                             for qa_type, color in color_map.items()]
        
        # Create edge method legend
        edge_legend_elements = [plt.Line2D([0], [0], color=method_color_map[method], lw=2, label=method)
                              for method in self.methods]
        
        # Combine legends
        all_legend_elements = node_legend_elements + edge_legend_elements
        plt.legend(handles=all_legend_elements, loc='upper right')
        
        # Save the figure
        plt.savefig(output_file)
        plt.close()
        
        print(f"Combined graph saved to {output_file}")
    
    def get_graph_statistics(self):
        """
        Get statistics about all graphs.
        
        Returns:
            Dictionary with statistics for each method
        """
        stats = {}
        
        for method in self.methods:
            # Convert adjacency list to NetworkX directed graph
            G = nx.DiGraph()
            
            # Add all nodes first
            for qa_pair in self.dataset:
                G.add_node(qa_pair.id)
            
            # Add edges
            for source_id, targets in self.adjacency_lists[method].items():
                for target_id in targets:
                    G.add_edge(source_id, target_id)
            
            # Calculate statistics
            method_stats = {
                "num_edges": G.number_of_edges(),
                "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
                "density": nx.density(G),
                "strongly_connected_components": nx.number_strongly_connected_components(G),
                "weakly_connected_components": nx.number_weakly_connected_components(G)
            }
            
            # Try to calculate these metrics if the graph is not empty
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                try:
                    method_stats["avg_clustering"] = nx.average_clustering(G)
                except:
                    method_stats["avg_clustering"] = "N/A"
                    
                try:
                    method_stats["diameter"] = nx.diameter(G.to_undirected())
                except:
                    method_stats["diameter"] = "N/A"
            else:
                method_stats["avg_clustering"] = "N/A"
                method_stats["diameter"] = "N/A"
            
            stats[method] = method_stats
            
        return stats


# Example usage:
if __name__ == "__main__":
    from base import QADataset, QAPair
    
    # Create sample dataset
    qa1 = QAPair(context="Deep learning is a subfield of machine learning.", 
                 question="What is deep learning?", 
                 answer="It's a subfield of machine learning.")
    
    qa2 = QAPair(context="Neural networks are used in deep learning.", 
                 question="What are neural networks used for?", 
                 answer="They are used in deep learning.")
    
    qa3 = QAPair(context="Python is a programming language.", 
                 question="What is Python?", 
                 answer="It's a programming language.")
    
    # Set IDs manually for this example
    qa1.id = 1
    qa2.id = 2
    qa3.id = 3
    
    dataset = QADataset(name="Sample dataset", qa_pairs=[qa1, qa2, qa3])
    
    # Create EdgeBuilder
    edge_builder = EdgeBuilder(dataset)
    
    # Build edges using cosine similarity
    edges_created = edge_builder.build_cosine_similarity_edges(threshold=0.3)
    print(f"Created {edges_created} edges using cosine similarity")
    
    # Build edges using keyword overlap
    edges_created = edge_builder.build_keyword_overlap_edges(threshold=1)
    print(f"Created {edges_created} edges using keyword overlap")
    
    # Manually add some edges with a custom method
    edge_builder.add_edge(1, 3, "manual")
    edge_builder.add_edge(3, 2, "manual")
    
    # Get all methods
    print(f"Available edge methods: {edge_builder.get_all_methods()}")
    
    # Visualize graphs
    edge_builder.visualize_all_graphs()
    edge_builder.visualize_combined_graph()
    
    # Print graph statistics
    stats = edge_builder.get_graph_statistics()
    for method, method_stats in stats.items():
        print(f"\nStatistics for method '{method}':")
        for stat_name, stat_value in method_stats.items():
            print(f"  {stat_name}: {stat_value}")