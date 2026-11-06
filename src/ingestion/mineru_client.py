import os
import time
import zipfile
from pathlib import Path
from typing import Dict

import requests
from dotenv import load_dotenv

requests.packages.urllib3.disable_warnings()
load_dotenv()


class MinerUClient:
    BASE_URL = "https://mineru.net/api/v4"

    def __init__(self):
        self.token = os.getenv("MINERU_API_TOKEN")
        if not self.token:
            raise ValueError("未配置 MINERU_API_TOKEN，请在 .env 中设置。")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
            "Accept": "*/*",
        }

    def upload_and_wait(
        self,
        pdf_path: str,
        output_dir: str = "data/processed",
        timeout: int = 300,
        poll_interval: int = 5,
        debug: bool = True,
        retry_upload: int = 3,
    ) -> Dict[str, str]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[MinerU] 开始上传: {pdf_path.name}")

        batch_id = None
        upload_success = False
        retry_count = 0

        while retry_count < retry_upload and not upload_success:
            try:
                batch_response = self._request_upload_url(pdf_path.name, debug=debug)
                batch_id = batch_response["batch_id"]
                upload_url = batch_response["file_urls"][0]

                print(f"[MinerU] 获得上传URL，batch_id={batch_id}")
                self._upload_file(pdf_path, upload_url, debug=debug)
                upload_success = True
                time.sleep(2)

            except Exception as e:
                retry_count += 1
                print(f"[MinerU] 上传失败（重试 {retry_count}/{retry_upload}）: {e}")
                if retry_count >= retry_upload:
                    raise RuntimeError(f"上传失败，已重试{retry_upload}次: {e}")
                time.sleep(2)

        result = self._poll_batch_status(batch_id, pdf_path.name, timeout, poll_interval, debug=debug)
        if result["state"] != "done":
            raise RuntimeError(f"MinerU解析失败: {result.get('err_msg', 'unknown')}")

        zip_url = result.get("full_zip_url")
        if not zip_url:
            raise RuntimeError("解析成功但未返回 zip 下载链接")

        zip_path = output_dir / f"{pdf_path.stem}_mineru.zip"
        self._download_file(zip_url, zip_path, debug=debug)

        json_path, md_path = self._extract_zip(zip_path, output_dir, pdf_path.stem, debug=debug)
        zip_path.unlink()

        print(f"[MinerU] ✅ 解析完成")
        print(f"  JSON: {json_path}")
        print(f"  MD:   {md_path}")

        return {
            "task_id": batch_id,
            "json_path": str(json_path),
            "md_path": str(md_path),
            "status": "success",
        }

    def _request_upload_url(self, filename: str, debug: bool = False) -> Dict:
        url = f"{self.BASE_URL}/file-urls/batch"
        data = {
            "files": [{"name": filename, "data_id": filename}],
            "model_version": "vlm",
            "enable_formula": True,
            "enable_table": True,
            "language": "ch",
        }

        response = requests.post(
            url,
            headers=self.headers,
            json=data,
            timeout=30,
            verify=False,
        )
        if debug:
            print(f"[DEBUG] 请求上传URL: {url}")
            print(f"[DEBUG] 响应状态: {response.status_code}")
            print(f"[DEBUG] 响应内容: {response.text[:500]}")

        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"申请上传URL失败: {result.get('msg', 'unknown error')}")
        return result["data"]

    def _upload_file(self, pdf_path: Path, upload_url: str, debug: bool = False):
        with open(pdf_path, "rb") as f:
            response = requests.put(
                upload_url,
                data=f,
                timeout=120,
                verify=False,
            )
        if debug:
            print(f"[DEBUG] 上传状态: {response.status_code}")
            print(f"[DEBUG] 上传结果: {response.text[:200]}")

        response.raise_for_status()

    def _poll_batch_status(
        self,
        batch_id: str,
        filename: str,
        timeout: int,
        interval: int,
        debug: bool = False,
    ) -> Dict:
        url = f"{self.BASE_URL}/extract-results/batch/{batch_id}"
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"MinerU解析超时：{timeout}s")

            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=30,
                    verify=False,
                )

                if response.status_code == 404:
                    time.sleep(interval)
                    continue

                if debug:
                    print(f"[DEBUG] 轮询URL: {url}")
                    print(f"[DEBUG] 状态码: {response.status_code}")
                    print(f"[DEBUG] 已等待: {elapsed:.1f}s")

                response.raise_for_status()
                result = response.json()

                if result.get("code") != 0:
                    if result.get("msg") and "not found" in str(result.get("msg")).lower():
                        time.sleep(interval)
                        continue
                    raise RuntimeError(f"任务查询失败: {result.get('msg', 'unknown')}")

                data = result.get("data", {})
                extract_result = data.get("extract_result") or []
                file_result = None
                for item in extract_result:
                    if item.get("file_name") == filename:
                        file_result = item
                        break

                if file_result is None:
                    time.sleep(interval)
                    continue

                state = file_result.get("state")
                err_msg = file_result.get("err_msg", "")
                if state == "done":
                    return file_result
                if state == "failed":
                    raise RuntimeError(f"MinerU解析失败: {err_msg}")

                if debug:
                    print(f"[DEBUG] 任务状态: {state} | 已等待: {elapsed:.1f}s")
                time.sleep(interval)

            except Exception as e:
                if debug:
                    print(f"[DEBUG] 轮询异常: {e}")
                time.sleep(interval)

    def _download_file(self, url: str, save_path: Path, debug: bool = False):
        try:
            response = requests.get(
                url,
                timeout=120,
                verify=False,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            )
            response.raise_for_status()
            save_path.write_bytes(response.content)
        except Exception as e:
            if debug:
                print(f"[DEBUG] requests 下载失败，回退到 curl: {e}")
            import subprocess
            proc = subprocess.run(
                ["curl", "-k", "-L", "--fail", "-o", str(save_path), url],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"下载失败: {proc.stderr or proc.stdout or str(e)}")
        if debug:
            print(f"[DEBUG] 下载完成: {save_path}")

    def _extract_zip(self, zip_path: Path, output_dir: Path, stem: str, debug: bool = False):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)

        def _best_match(candidates, wanted: str, suffix: str):
            preferred = []
            for p in candidates:
                name = p.name.lower()
                if name.startswith(f"{wanted.lower()}.") or name.startswith(f"{wanted.lower()}_"):
                    preferred.append(p)
            if preferred:
                return sorted(preferred, key=lambda p: p.name.lower())
            return sorted(candidates, key=lambda p: p.name.lower())

        json_candidates = sorted(output_dir.rglob("*_model.json"))
        if not json_candidates:
            json_candidates = sorted(output_dir.rglob("*.json"))
        md_candidates = sorted(output_dir.rglob("*.md"))
        if not md_candidates:
            md_candidates = sorted(output_dir.rglob("**/*.md"))

        if not json_candidates:
            raise FileNotFoundError(f"未找到 JSON 文件，zip: {zip_path}")
        if not md_candidates:
            raise FileNotFoundError(f"未找到 MD 文件，zip: {zip_path}")

        json_candidates = _best_match(json_candidates, stem, ".json")
        md_candidates = _best_match(md_candidates, stem, ".md")

        chosen_json = json_candidates[0]
        chosen_md = md_candidates[0]

        final_json = output_dir / f"{stem}.json"
        final_md = output_dir / f"{stem}.md"

        if chosen_json != final_json:
            if final_json.exists():
                final_json.unlink()
            chosen_json.rename(final_json)
        if chosen_md != final_md:
            if final_md.exists():
                final_md.unlink()
            chosen_md.rename(final_md)

        if debug:
            print(f"[DEBUG] 找到 JSON: {[str(p) for p in json_candidates[:5]]}")
            print(f"[DEBUG] 找到 MD: {[str(p) for p in md_candidates[:5]]}")
            print(f"[DEBUG] 已规范化为: {final_json}, {final_md}")

        return str(final_json), str(final_md)
