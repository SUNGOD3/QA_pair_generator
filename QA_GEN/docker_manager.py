#docker_manager.py
import docker
import yaml
import tempfile
import shutil
from typing import Dict, Any, List
from pathlib import Path
import pickle

class DockerMethodManager:
    """
    管理Docker環境中的方法執行
    """
    
    def __init__(self):
        """
        初始化Docker方法管理器
        
        Args:
            base_image: 基礎Docker鏡像
        """
        self.client = docker.from_env()
        self.base_image = "python:3.9-slim"
        self.method_configs_dir = Path("QA_GEN/method_configs")
        self.temp_dirs = []  # 追蹤臨時目錄以便清理
        
        # 確保配置目錄存在
        self.method_configs_dir.mkdir(exist_ok=True)
        
    def load_method_config(self, method_name: str) -> Dict[str, Any]:
        """
        載入方法的配置檔案
        
        Args:
            method_name: 方法名稱
            
        Returns:
            方法配置字典
        """
        config_path = self.method_configs_dir / f"{method_name}.yaml"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Loaded config for {method_name} from {config_path}")
        else:
            raise FileNotFoundError(f"Configuration file for method '{method_name}' not found at {config_path}")
            
        return config
    
    def _create_dockerfile(self, config: Dict[str, Any], temp_dir: Path) -> str:
        """
        根據配置創建Dockerfile
        
        Args:
            config: 方法配置
            temp_dir: 臨時目錄路徑
            
        Returns:
            Dockerfile內容
        """
        base_image = config.get('base_image', self.base_image)
        dependencies = config.get('dependencies', [])
        env_vars = config.get('environment_variables', {})
        workdir = config.get('working_directory', '/app')
        
        dockerfile_content = f"""FROM {base_image}

WORKDIR {workdir}

# 安裝系統依賴
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# 安裝Python依賴
"""
        
        if dependencies:
            deps_str = ' '.join(dependencies)
            dockerfile_content += f"RUN pip install --no-cache-dir {deps_str}\n"
        
        # 設置環境變數
        for key, value in env_vars.items():
            dockerfile_content += f"ENV {key}={value}\n"
        
        dockerfile_content += """
# 複製執行腳本
COPY . .

# 執行方法的入口點
CMD ["python", "execute_method.py"]
"""
        
        return dockerfile_content
    
    def _create_execution_script(self, method_name: str, temp_dir: Path):
        """
        創建方法執行腳本
        
        Args:
            method_name: 方法名稱
            temp_dir: 臨時目錄路徑
        """
        script_content = f'''
import pickle
import sys
import os
import traceback


# 設置Python路徑
sys.path.insert(0, '/app')

import QA_GEN.base as base_module
import QA_GEN.methods as methods_module


def main():
    try:
        # load QA pairs and config
        with open('/app/input_data.pkl', 'rb') as f:
            raw_data = pickle.load(f)
        
        if isinstance(raw_data, tuple) and len(raw_data) == 2:
            qa_pairs, config = raw_data
        else:
            qa_pairs = raw_data
            config = {{}}
        
        print(f"Config: {{config}}")
        print(f"QA pairs count: {{len(qa_pairs)}}")
        
        # execute the specified method
        methods = methods_module.Method.get_methods()
        print(f"Available methods: {{list(methods.keys())}}")
        
        if "{method_name}" not in methods:
            raise ValueError(f"Method {method_name} not found. Available: {{list(methods.keys())}}")
        
        method_func = methods["{method_name}"]["func"]
        print(f"Executing method: {method_name}")
        
        result = method_func(qa_pairs, config)
        
        print(f"Method execution completed. Result type: {{type(result)}}")
        if hasattr(result, '__len__'):
            print(f"Result length: {{len(result)}}")
        
        # save the result
        with open('/app/output_data.pkl', 'wb') as f:
            pickle.dump(result, f)
            
        print(f"Method {method_name} executed successfully")
        
    except Exception as e:
        print(f"Error executing method {method_name}: {{str(e)}}")
        print("Full traceback:")
        traceback.print_exc()
        
        # output error to a file
        with open('/app/error.txt', 'w') as f:
            f.write(f"{{str(e)}}\\n\\nFull traceback:\\n{{traceback.format_exc()}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        script_path = temp_dir / "execute_method.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
    
    def execute_method_in_docker(self, method_name: str, qa_pairs: List, config: Dict[str, Any]) -> List:
        """
        在Docker環境中執行方法
        
        Args:
            method_name: 方法名稱
            qa_pairs: QA對列表
            config: 配置參數
            
        Returns:
            執行結果
        """
        print(f"Executing method '{method_name}' in Docker environment...")
        
        # 載入方法配置
        method_config = self.load_method_config(method_name)
        
        # 創建臨時目錄
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        
        try:
            # 創建Dockerfile
            dockerfile_content = self._create_dockerfile(method_config, temp_dir)
            dockerfile_path = temp_dir / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # 創建執行腳本
            self._create_execution_script(method_name, temp_dir)
            
            # 複製必要的Python文件
            self._copy_python_files(temp_dir)
            
            # 準備並保存輸入數據
            input_data = qa_pairs 
            with open(temp_dir / "input_data.pkl", 'wb') as f:
                pickle.dump(input_data, f)
            
            # 構建Docker鏡像
            image_tag = f"qa_method_{method_name.lower()}"
            print(f"Building Docker image: {image_tag}")
            image, build_logs = self.client.images.build(
                path=str(temp_dir),
                tag=image_tag,
                rm=True
            )
            
            # 運行容器
            print(f"Running container for method: {method_name}")
            container = self.client.containers.run(
                image_tag,
                detach=True,
                volumes={str(temp_dir): {'bind': '/app', 'mode': 'rw'}},
                working_dir='/app'
            )
            
            # 等待容器完成
            result = container.wait(timeout=method_config.get('timeout', 300))
            
            # 獲取日誌
            logs = container.logs().decode('utf-8')
            print(f"Container logs:\n{logs}")
            
            # 清理容器
            container.remove()
            
            # 檢查執行結果
            if result['StatusCode'] != 0:
                error_file = temp_dir / "error.txt"
                if error_file.exists():
                    with open(error_file, 'r') as f:
                        error_msg = f.read()
                    raise RuntimeError(f"Method execution failed: {error_msg}")
                else:
                    raise RuntimeError(f"Method execution failed with status code: {result['StatusCode']}")
            
            # 載入結果
            output_file = temp_dir / "output_data.pkl"
            if output_file.exists():
                with open(output_file, 'rb') as f:
                    raw_result = pickle.load(f)
                return raw_result
            else:
                raise RuntimeError("No output data found")
                
        finally:
            # 清理Docker鏡像
            try:
                self.client.images.remove(image_tag, force=True)
            except:
                pass
    
    def _copy_python_files(self, temp_dir: Path):
        """
        複製必要的Python文件到臨時目錄
        
        Args:
            temp_dir: 臨時目錄路徑
        """
        # 將整個QA_GEN目錄下的所有文件和資料夾複製到臨時目錄
        qa_gen_path = Path('QA_GEN')
        if qa_gen_path.exists():
            for item in qa_gen_path.rglob('*'):
                if item.is_file() and not item.name.startswith('.'):
                    # 計算相對路徑並直接放在temp_dir下（扁平化結構）
                    relative_path = item.relative_to(qa_gen_path)
                    
                    # 如果是在子目錄中的文件，創建對應的子目錄結構
                    dest_path = temp_dir / "QA_GEN" / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        shutil.copy2(item, dest_path)
                    except Exception as e:
                        print(f"Warning: Could not copy {item}: {e}")
        else:
            print("Warning: QA_GEN directory not found!")
    
    def cleanup(self):
        """
        清理臨時目錄
        """
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
    
    def __del__(self):
        """
        析構函數，自動清理資源
        """
        self.cleanup()