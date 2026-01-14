from bs4 import BeautifulSoup, NavigableString, Tag
from apted import APTED, Config
import lxml

class TableTree(Config):
    def rename(self, node1, node2):
        """Compares two nodes label for equality."""
        return 1 if node1.name != node2.name else 0

    def children(self, node):
        """Return list of children for a node."""
        return [child for child in node.children if isinstance(child, Tag)]

class TEDS:
    def __init__(self):
        pass

    def html_to_tree(self, html_content):
        """Parses HTML and returns the root element (skipping <html><body> if present)."""
        soup = BeautifulSoup(html_content, 'lxml')
        # Find the table tag
        table = soup.find('table')
        if table:
            return table
        # If no table tag, return the soup itself if it has children, else None
        if soup.body:
            return soup.body
        return soup

    def evaluate(self, pred_html, gt_html):
        """
        Calculates TEDS score between prediction and ground truth.
        Score = 1 - (EditDistance / Max(|Tree1|, |Tree2|))
        """
        try:
            tree_pred = self.html_to_tree(pred_html)
            tree_gt = self.html_to_tree(gt_html)

            if tree_pred is None or tree_gt is None:
                return 0.0

            # Calculate Tree Edit Distance using APTED
            apted = APTED(tree_pred, tree_gt, TableTree())
            ted = apted.compute_edit_distance()
            
            # Count nodes for normalization
            count_pred = self.count_nodes(tree_pred)
            count_gt = self.count_nodes(tree_gt)
            
            max_nodes = max(count_pred, count_gt)
            if max_nodes == 0:
                return 0.0
                
            score = 1.0 - (ted / max_nodes)
            return max(0.0, score)
            
        except Exception as e:
            print(f"TEDS calculation error: {e}")
            return 0.0

    def count_nodes(self, node):
        count = 1
        for child in node.children:
            if isinstance(child, Tag):
                count += self.count_nodes(child)
        return count
