import json

class ActionExecutor:
    @staticmethod
    def convert_to_json(cls):
        """
        将工具类转换为 JSON 格式
        """
        properties = {}
        required = []
        for param in cls.parameters:
            properties[param['name']] = {
                'type': param['type'],
                'description': param['description']
            }
            if param.get('required', False):
                required.append(param['name'])

        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    @staticmethod
    def get_tool_info(*classes):
        """
        转换多个工具类为 JSON 格式并返回
        """
        tool_info = []
        tool_map = {}
        for cls in classes:
            json_data = ActionExecutor.convert_to_json(cls)
            tool_info.append(json_data)
            tool_map[cls.name] = cls
        return tool_info, tool_map
