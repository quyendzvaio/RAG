from shared.config import get_settings


if __name__ == "__main__":
    settings = get_settings().deepseek
    print("DEEPSEEK_BASE_URL:", settings.base_url)
    print("DEEPSEEK_MODEL:", settings.model)
    print("DEEPSEEK_API_KEY configured:", settings.configured)
