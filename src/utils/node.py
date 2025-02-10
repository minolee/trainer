
__all__ = ["parse_nodelist", "is_localhost"]
from socket import gethostname
def parse_nodelist(nodelist: str) -> list:
    """
    Slurm의 nodelist 문자열을 파싱하여 실제 노드 이름 리스트로 반환합니다.
    (ChatGPT generated)
    
    예제:
        "node[01-03,05]" → ["node01", "node02", "node03", "node05"]
        "node[001-003]-gpu" → ["node001-gpu", "node002-gpu", "node003-gpu"]
        "nodeA,nodeB,node[1-2]" → ["nodeA", "nodeB", "node1", "node2"]
    
    Args:
        nodelist (str): 파싱할 nodelist 문자열.
    
    Returns:
        list: 확장된 노드 이름들의 리스트.
    """
    
    def split_top_level(s: str) -> list:
        """
        괄호 안의 콤마는 무시하고 top-level에서만 콤마로 문자열을 분리합니다.
        """
        parts = []
        current = []
        depth = 0
        for char in s:
            if char == '[':
                depth += 1
                current.append(char)
            elif char == ']':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)
        if current:
            parts.append(''.join(current))
        return parts

    def expand(expr: str) -> list:
        """
        expr에 괄호 확장이 있다면 재귀적으로 확장하여 모든 조합을 반환합니다.
        """
        # 더 이상 확장할 괄호가 없으면 그대로 반환.
        if '[' not in expr:
            return [expr]
        
        # 가장 왼쪽의 '['를 찾아서, 그에 대응하는 ']' 위치를 찾음.
        i = expr.find('[')
        depth = 0
        for j in range(i, len(expr)):
            if expr[j] == '[':
                depth += 1
            elif expr[j] == ']':
                depth -= 1
                if depth == 0:
                    break
        else:
            raise ValueError("Unmatched '[' in expression: " + expr)
        
        # expr는 "prefix[inside]suffix" 형태로 가정.
        prefix = expr[:i]
        inside = expr[i + 1 : j]
        suffix = expr[j + 1 :]
        
        result = []
        # inside 내부의 항목들은 콤마로 구분되어 있다.
        for part in inside.split(','):
            # part가 범위 (예: "01-03")인지 혹은 단일 값 (예: "05")인지 확인.
            if '-' in part:
                start_str, end_str = part.split('-', 1)
                try:
                    start = int(start_str)
                    end = int(end_str)
                except ValueError:
                    # 만약 숫자가 아니면 그냥 그대로 처리합니다.
                    values = [part]
                else:
                    width = len(start_str)  # 숫자의 자릿수를 유지하기 위함.
                    values = [str(n).zfill(width) for n in range(start, end + 1)]
            else:
                values = [part]
            
            for value in values:
                # 접미사(suffix)에 괄호가 있을 수 있으므로 재귀 호출.
                for tail in expand(suffix):
                    result.append(prefix + value + tail)
        return result

    final_nodes = []
    # nodelist 전체를 top-level 콤마 기준으로 분리한 후 각각 확장.
    for part in split_top_level(nodelist):
        final_nodes.extend(expand(part))
    return final_nodes


def is_localhost(host):
    return host == "localhost" or host == gethostname()