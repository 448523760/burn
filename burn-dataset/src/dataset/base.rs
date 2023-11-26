use std::sync::Arc;

use crate::DatasetIterator;

/// The dataset trait defines a basic collection of items with a predefined size.
pub trait Dataset<I>: Send + Sync {
    /// Gets the item at the given index.
    fn get(&self, index: usize) -> Option<I>;

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    fn iter(&self) -> DatasetIterator<'_, I>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

///
/// 这是对 Dataset<I> trait 的实现部分，实现类型是 Arc<D>。
/// Arc<D> 是 Rust 中的智能指针类型，它允许多个所有者共享数据，并提供线程安全的引用计数。
/// 在这里，我们将 Arc<D> 作为数据集的类型。
///
/// D 是一个泛型参数，表示实现了 Dataset<I> trait 的具体数据集类型。
///
/// where D: Dataset<I> 是一个约束条件，它要求 D 类型必须实现 Dataset<I> trait。
/// 这意味着我们只能为实现了 Dataset<I> trait 的数据集类型 D 提供这个实现。
///
///
///
impl<D, I> Dataset<I> for Arc<D>
where
    D: Dataset<I>,
{
    ///
    /// 在这个方法中，我们通过 self.as_ref() 将 Arc<D> 转换为 &D 的引用，
    /// 然后调用 get 方法来获取具体数据集类型 D 中给定索引的项目。
    ///
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}


/// Arc<dyn Dataset<I>> 表示一个动态多态类型，它可以持有任何实现了 Dataset<I> trait 的具体类型。
impl<I> Dataset<I> for Arc<dyn Dataset<I>> {
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<D, I> Dataset<I> for Box<D>
where
    D: Dataset<I>,
{
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I> Dataset<I> for Box<dyn Dataset<I>> {
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}
