"""Routes for toolkits and tool execution."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Form, File, UploadFile, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..db import models as m
from .factory import RouterConfig, SDK_CONFIG, GUI_CONFIG
from .deps import current_user_from_api_key, current_user_from_jwt
from ..core.schemas import (
    ToolSpecOut, ToolkitOut, ExecuteRequestIn, ExecuteResponseOut,
)
from ..core.services import ToolService
from ..core.utils.uploads import save_upload_files
from ..core.utils.runtime_paths import prepare_runtime_context, runtime_root
from ..core.utils.tool_frontend_schema import prepare_gui_tool_parameters

import json


# Business logic helpers (shared between SDK/GUI)

def _get_tools_with_status(db: Session, user_id: str) -> list[ToolSpecOut]:
    return ToolService.get_tools_with_status(db, user_id)


def _get_tool_with_status_or_404(db: Session, user_id: str, slug: str) -> ToolSpecOut:
    tool = ToolService.get_tool_with_status(db, user_id, slug)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found"
        )
    return tool


def _get_toolkits_with_status(db: Session, user_id: str) -> list[ToolkitOut]:
    return ToolService.get_toolkits_with_status(db, user_id)


async def _execute_tool(db: Session, user_id: str, slug: str, request: ExecuteRequestIn) -> ExecuteResponseOut:
    return await ToolService.execute_tool(db, user_id, slug, request)


def _resolve_slug(slug: str, db: Session, user_id: str) -> str:
    """Resolve a potentially underscored slug (from LLM) back to a dotted slug."""
    if "." in slug:
        return slug
    
    # Get all tools to find a match
    tools = _get_tools_with_status(db, user_id)
    for tool in tools:
        if tool.slug.replace(".", "_") == slug:
            return tool.slug
            
    return slug


def _metadata_list(value: Any) -> list[str]:
    """Parse a metadata value that may contain a JSON or comma-separated list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = [part.strip() for part in value.split(",")]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
    return []


def _assign_uploaded_path(inputs: Dict[str, Any], param_name: str, path: str) -> None:
    if "." in param_name:
        head, *tail = [part for part in param_name.split(".") if part]
        if not head or not tail:
            return
        current = inputs.get(head)
        if not isinstance(current, dict):
            current = {}
            inputs[head] = current
        for part in tail[:-1]:
            if not isinstance(current, dict):
                return
            existing = current.get(part)
            if not isinstance(existing, dict):
                existing = {}
                current[part] = existing
            current = existing
        if isinstance(current, dict):
            current[tail[-1]] = path
        return

    current = inputs.get(param_name)
    if isinstance(current, list):
        current.append(path)
    elif param_name.endswith("s") or param_name.endswith("_paths"):
        inputs[param_name] = [path] if current in (None, "") else [current, path]
    else:
        inputs[param_name] = path


def _map_uploaded_files_to_inputs(
    inputs: Dict[str, Any],
    saved_paths: list[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Map uploaded files to explicit tool parameters, with image compatibility fallback."""
    mapped = dict(inputs)
    param_names = _metadata_list(metadata.get("file_param_names"))

    if param_names:
        for index, path in enumerate(saved_paths):
            param_name = param_names[index] if index < len(param_names) else param_names[-1]
            _assign_uploaded_path(mapped, param_name, path)
        return mapped

    if len(saved_paths) > 1:
        mapped["images"] = saved_paths
        mapped["image_paths"] = saved_paths
    elif saved_paths:
        mapped["image"] = saved_paths[0]
        mapped["images"] = saved_paths
        mapped["image_path"] = saved_paths[0]
        mapped.setdefault("image_paths", saved_paths)
    return mapped


def _resolve_runtime_file_path(file_path: str) -> Path:
    """Resolve and validate a runtime file path before returning it to the client."""
    root = runtime_root().resolve()
    requested = Path(file_path).expanduser().resolve()
    try:
        requested.relative_to(root)
    except ValueError as exc:
        raise ValueError("Requested file is outside the Terrabox runtime directory") from exc
    if not requested.exists() or not requested.is_file():
        raise FileNotFoundError(file_path)
    return requested


def _prepare_tool_for_gui(tool: ToolSpecOut) -> ToolSpecOut:
    parameters, gui_metadata = prepare_gui_tool_parameters(tool.slug, tool.parameters)
    if not gui_metadata:
        return tool
    metadata = dict(tool.metadata or {})
    metadata.update(gui_metadata)
    return tool.model_copy(
        deep=True,
        update={
            "parameters": parameters,
            "metadata": metadata,
        },
    )


def _prepare_toolkit_for_gui(toolkit: ToolkitOut) -> ToolkitOut:
    return toolkit.model_copy(
        deep=True,
        update={
            "tools": [_prepare_tool_for_gui(tool) for tool in toolkit.tools],
        },
    )


# =============================================================================
# Router Factory Implementation
# =============================================================================

def make_tools_router(config: RouterConfig) -> APIRouter:
    """Create a tools router with the specified configuration.
    
    Args:
        config: Router configuration (prefix, auth dependency, etc.)
        
    Returns:
        Configured APIRouter with all tools endpoints
    """
    router = APIRouter(
        prefix=config.prefix,
        tags=[f"tools-{config.prefix.split('/')[-1]}"]
    )
    
    @router.get("/tools", response_model=list[ToolSpecOut])
    def get_tools(
        toolkit: str | None = None,
        current_user: m.User = Depends(config.current_user_dep),
        db: Session = Depends(get_db)
    ):
        """Get all tools with their availability status."""
        tools = _get_tools_with_status(db, current_user.user_id)
        if config.prefix == "/v1/gui":
            return [_prepare_tool_for_gui(tool) for tool in tools]
        return tools

    @router.get("/tools/definitions/openai", response_model=List[Dict[str, Any]])
    def get_tools_openai_format(
        toolkit: str | None = None,
        current_user: m.User = Depends(config.current_user_dep),
        db: Session = Depends(get_db)
    ):
        """Get tools in OpenAI function calling format."""
        tools = _get_tools_with_status(db, current_user.user_id)
        if toolkit:
            tools = [t for t in tools if t.toolkit_slug == toolkit]
            
        openai_tools = []
        for tool in tools:
            if tool.status != "available": continue
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.slug.replace(".", "_"),
                    "description": tool.description,
                    "parameters": tool.parameters or {"type": "object", "properties": {}}
                }
            })
        return openai_tools

    @router.get("/tools/definitions/anthropic", response_model=List[Dict[str, Any]])
    def get_tools_anthropic_format(
        toolkit: str | None = None,
        current_user: m.User = Depends(config.current_user_dep),
        db: Session = Depends(get_db)
    ):
        """Get tools in Anthropic tool use format."""
        tools = _get_tools_with_status(db, current_user.user_id)
        if toolkit:
            tools = [t for t in tools if t.toolkit_slug == toolkit]
            
        anthropic_tools = []
        for tool in tools:
            if tool.status != "available": continue
            anthropic_tools.append({
                "name": tool.slug.replace(".", "_"),
                "description": tool.description,
                "input_schema": tool.parameters or {"type": "object", "properties": {}}
            })
        return anthropic_tools

    @router.get("/tools/definitions/gemini", response_model=List[Dict[str, Any]])
    def get_tools_gemini_format(
        toolkit: str | None = None,
        current_user: m.User = Depends(config.current_user_dep),
        db: Session = Depends(get_db)
    ):
        """Get tools in Google Gemini format."""
        tools = _get_tools_with_status(db, current_user.user_id)
        if toolkit:
            tools = [t for t in tools if t.toolkit_slug == toolkit]
            
        funcs = []
        for tool in tools:
            if tool.status != "available": continue
            funcs.append({
                "name": tool.slug.replace(".", "_"),
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}}
            })
        return [{"function_declarations": funcs}]

    @router.get("/tools/{slug}", response_model=ToolSpecOut)
    def get_tool_detail(
        slug: str,
        current_user: m.User = Depends(config.current_user_dep),
        db: Session = Depends(get_db)
    ):
        """Get a specific tool with its availability status."""
        tool = _get_tool_with_status_or_404(db, current_user.user_id, slug)
        if config.prefix == "/v1/gui":
            return _prepare_tool_for_gui(tool)
        return tool

    @router.get("/toolkits", response_model=list[ToolkitOut])
    def get_toolkits(
        current_user: m.User = Depends(config.current_user_dep),
        db: Session = Depends(get_db)
    ):
        """Get all toolkits with their tools' availability status."""
        toolkits = _get_toolkits_with_status(db, current_user.user_id)
        if config.prefix == "/v1/gui":
            return [_prepare_toolkit_for_gui(toolkit) for toolkit in toolkits]
        return toolkits

    @router.post("/tools/{slug}/execute", response_model=ExecuteResponseOut)
    async def execute_tool(
        slug: str,
        request: ExecuteRequestIn,
        current_user: m.User = Depends(config.current_user_dep),
        db: Session = Depends(get_db)
    ):
        """Execute a tool."""
        real_slug = _resolve_slug(slug, db, current_user.user_id)
        return await _execute_tool(db, current_user.user_id, real_slug, request)

    @router.post("/tools/{slug}/execute-multipart", response_model=ExecuteResponseOut)
    async def execute_tool_multipart(
        slug: str,
        inputs: str = Form(...),
        metadata: Optional[str] = Form(None),
        files: List[UploadFile] = File(...),
        current_user: m.User = Depends(config.current_user_dep),
        db: Session = Depends(get_db),
    ):
        """Execute a tool with file uploads (multipart/form-data)."""
        try:
            inputs_dict: Dict[str, Any] = json.loads(inputs or "{}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid inputs JSON: {e}")

        metadata_dict: Dict[str, Any] = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except Exception:
                metadata_dict = {}

        real_slug = _resolve_slug(slug, db, current_user.user_id)
        runtime_metadata = prepare_runtime_context(
            current_user.user_id,
            real_slug,
            execution_id=metadata_dict.get("execution_id"),
        )
        metadata_dict.update(runtime_metadata)

        saved_paths = await save_upload_files(files, upload_dir=runtime_metadata["upload_dir"])

        inputs_dict = _map_uploaded_files_to_inputs(inputs_dict, saved_paths, metadata_dict)

        request_in = ExecuteRequestIn(inputs=inputs_dict, metadata=metadata_dict or None)
        return await _execute_tool(db, current_user.user_id, real_slug, request_in)

    @router.get("/runtime/files")
    def download_runtime_file(
        path: str = Query(..., description="Absolute path under TERRABOX_RUNTIME_DIR."),
        current_user: m.User = Depends(config.current_user_dep),
    ):
        """Download a file produced under the Terrabox runtime directory."""
        try:
            resolved = _resolve_runtime_file_path(path)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Runtime file not found") from exc
        return FileResponse(resolved, filename=resolved.name)

    return router

# =============================================================================
# Create SDK and GUI routers using factory
# =============================================================================

# Create SDK router (API Key authentication)
sdk_router = make_tools_router(SDK_CONFIG)

# Create GUI router (JWT authentication)  
gui_router = make_tools_router(GUI_CONFIG)


# Export routers for main app
__all__ = ["sdk_router", "gui_router", "make_tools_router"]
