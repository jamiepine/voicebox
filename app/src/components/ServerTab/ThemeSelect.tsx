import { useTranslation } from 'react-i18next';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { type Theme, useUIStore } from '@/stores/uiStore';

export function ThemeSelect() {
  const { t } = useTranslation();
  const theme = useUIStore((s) => s.theme);
  const setTheme = useUIStore((s) => s.setTheme);

  return (
    <Select value={theme} onValueChange={(value) => setTheme(value as Theme)}>
      <SelectTrigger className="h-9 w-[180px]">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="system">{t('settings.theme.options.system')}</SelectItem>
        <SelectItem value="light">{t('settings.theme.options.light')}</SelectItem>
        <SelectItem value="dark">{t('settings.theme.options.dark')}</SelectItem>
      </SelectContent>
    </Select>
  );
}
